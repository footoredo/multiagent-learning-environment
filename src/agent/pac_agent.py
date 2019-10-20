from monitor.statistics import Statistics
from agent.base_agent import BaseAgent
from agent.policy import Policy
import numpy as np
import tensorflow as tf
import baselines.common.tf_util as U
from tensorflow.contrib import layers
from baselines.common.distributions import make_pdtype
from baselines.common.mpi_adam import MpiAdam
from baselines.common import Dataset
from mpi4py import MPI
import time
from common.path_utils import *


def make_mlp(scope, inputs, num_outputs, num_units, depth=2, reuse=False):
    last = inputs
    with tf.variable_scope(scope, reuse=reuse):
        for i in range(depth - 1):
            with tf.variable_scope("l_{}".format(i), reuse=reuse):
                last = layers.fully_connected(last, num_units, activation_fn=tf.nn.relu)
        with tf.variable_scope("l_last", reuse=reuse):
            output = layers.fully_connected(last, num_outputs, activation_fn=None)
    return output


class PACAgent(BaseAgent):
    def __init__(self, name, *args, **kwargs):
        self.name = name
        with tf.variable_scope(name):
            # print(name)
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def save(self, save_path):
        # json.dump(self.config, open(join_path_and_check(save_path, "config.json"), "w"))
        U.save_variables(join_path(save_path, "model.obj"),
                         variables=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name))

    def load(self, load_path):
        U.load_variables(join_path(load_path, "model.obj"),
                         variables=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name))

    def _init(self, policy_fn, ob_space, ac_space,
              handlers,
              timesteps_per_actorbatch,
              optim_epochs, optim_stepsize,  # optimization hypers
              gamma,
              max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
              adam_epsilon=1e-5,
              schedule='constant',  # annealing for stepsize parameters (epsilon and adam)
              opponent='latest',  # opponent type
              exploration=None):

        self.policy_fn = policy_fn
        self.ob_space = ob_space
        self.ac_space = ac_space
        self.push, self.pull = handlers
        self.pi = pi = policy_fn("pi", self.name, ob_space, ac_space)
        ac_pdtype = make_pdtype(ac_space)
        assert len(self.ob_space.shape) == 1
        self.n_ob = self.ob_space.shape[0]
        self.n_ac = ac_pdtype.ncat

        ob = U.get_placeholder_cached(name="ob_" + self.name)
        ac = tf.placeholder(dtype=tf.int32, shape=[None], name="ac")
        ac_one_hot = tf.one_hot(ac, depth=self.n_ac, axis=-1)
        q_input = tf.concat([ob, ac_one_hot], axis=1)
        q = make_mlp("q", q_input, num_outputs=1, num_units=64)[:, 0]

        self.q_func = U.function([ob, ac], q)

        q_ac = []
        for i in range(self.n_ac):
            ac_i = tf.one_hot(indices=i, depth=self.n_ac)
            batch_size = tf.shape(ob)[0]
            ac_i = tf.expand_dims(ac_i, 0)
            ac_i = tf.tile(ac_i, tf.stack([batch_size, 1]))
            q_input_i = tf.concat([ob, ac_i], axis=1)
            q_i = make_mlp("q", q_input_i, num_outputs=1, num_units=64, reuse=True)
            q_ac.append(q_i)

        q_ac = tf.stack(q_ac)  # n_ac * batch * 1
        q_ac = tf.transpose(q_ac, [1, 0, 2])  # batch * n_ac * 1
        ac_p = tf.nn.softmax(pi.pd.flatparam())  # batch * n_ac
        ac_p = tf.expand_dims(ac_p, 1)  # batch * 1 * n_ac

        loss_p = tf.reduce_mean(tf.matmul(ac_p, q_ac))

        target_ph = tf.placeholder(dtype=tf.float32, shape=[None], name="target")
        loss_q = tf.reduce_mean(tf.square(q - target_ph))

        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

        p_var_list = pi.get_trainable_variables()
        adam_p = MpiAdam(var_list=p_var_list, epsilon=adam_epsilon)

        q_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name + "/q")
        adam_q = MpiAdam(var_list=q_var_list, epsilon=adam_epsilon)

        self.calc_loss_p = U.function([ob], [loss_p, q_ac, U.flatgrad(loss_p, p_var_list)])
        self.calc_loss_q = U.function([ob, ac, target_ph], [loss_q, U.flatgrad(loss_q, q_var_list)])

        U.initialize()
        adam_p.sync()
        adam_q.sync()


        def train_phase(i, statistics: Statistics, progress):
            env = self.pull(opponent)
            seg_gen = self._traj_segment_generator(self.pi, self.q_func, env, timesteps_per_actorbatch, stochastic=True)

            episodes_so_far = 0
            timesteps_so_far = 0
            iters_so_far = 0
            tstart = time.time()
            # lenbuffer = deque(maxlen=100)  # rolling buffer for episode lengths
            # rewbuffer = deque(maxlen=100)  # rolling buffer for episode rewards

            assert sum([max_iters > 0, max_timesteps > 0, max_episodes > 0,
                        max_seconds > 0]) == 1, "Only one time constraint permitted"

            while True:
                # if callback: callback(locals(), globals())
                if max_timesteps and timesteps_so_far >= max_timesteps:
                    break
                elif max_episodes and episodes_so_far >= max_episodes:
                    break
                elif max_iters and iters_so_far >= max_iters:
                    break
                elif max_seconds and time.time() - tstart >= max_seconds:
                    break
                # print(max_iters, iters_so_far)

                # logger.log("********** Iteration %i ************" % iters_so_far)

                seg = seg_gen.__next__()
                self._add_q_target(seg, gamma)

                # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
                # logger.log(seg["rew"])
                ob, ac, qtarget = seg["ob"], seg["ac"], seg["qtarget"]
                # print(ob, ac, qtarget, seg["q"])
                # logger.log(atarg)
                # vpredbefore = seg["vpred"]  # predicted value function before udpate
                # if atarg.std() > 1e-4:
                #     atarg = (atarg - atarg.mean()) / atarg.std()  # standardized advantage function estimate
                # else:
                #     print(atarg.mean())
                #     atarg = (atarg - atarg.mean())
                # print(atarg)
                d = Dataset(dict(ob=ob, ac=ac, qtarget=qtarget), deterministic=False)
                optim_batchsize = ob.shape[0]
                # print("optim_batchsize:", optim_batchsize)

                # cur_lrmult *= 1e-2 / np.sqrt(progress + 1e-2)
                # print(1e-2 / np.sqrt(progress))

                # self.average_utility = self.average_utility * self.tot + np.sum(seg["rew"])
                # self.tot += len(seg["rew"])
                # self.average_utility /= self.tot

                # fasdaqwe = 0.9
                # self.average_utility = fasdaqwe * self.average_utility + np.average(seg["rew"]) * (1 - fasdaqwe)

                # print(self.scope + ("avg util: %.5f" % self.average_utility))

                # if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob)  # update running mean/std for policy

                # assign_old_eq_new()  # set old parameter values to new parameter values
                # logger.log("Optimizing...")
                # logger.log(fmt_row(13, loss_names))
                # Here we do a bunch of optimization epochs over the data
                for _ in range(optim_epochs):
                    # losses = []  # list of tuples, each of which gives the loss for a minibatch
                    for batch in d.iterate_once(optim_batchsize):
                        lq, gq = self.calc_loss_q(batch["ob"], batch["ac"], batch["qtarget"])
                        adam_q.update(gq, optim_stepsize)
                        # print("lq", lq)

                for _ in range(optim_epochs):
                    # losses = []  # list of tuples, each of which gives the loss for a minibatch
                    for batch in d.iterate_once(optim_batchsize):
                        lp, q_ac, gp = self.calc_loss_p(batch["ob"])
                        adam_p.update(-gp, optim_stepsize)
                        # print("lp", lp, q_ac[0])
                    #     losses.append(newlosses)
                    # logger.log(fmt_row(13, np.mean(losses, axis=0)))

                # logger.log("Evaluating losses...")
                # losses = []
                # for batch in d.iterate_once(optim_batchsize):
                #     newlosses = compute_losses(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                #     print(newlosses)
                #     losses.append(newlosses)
                # meanlosses, _, _ = mpi_moments(losses, axis=0)
                # logger.log(fmt_row(13, meanlosses))
                # for (lossval, name) in zipsame(meanlosses, loss_names):
                #     logger.record_tabular("loss_" + name, lossval)
                # logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))
                lrlocal = (seg["ep_lens"], seg["ep_rets"])  # local values
                listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal)  # list of tuples
                lens, rews = map(flatten_lists, zip(*listoflrpairs))
                # lenbuffer.extend(lens)
                # rewbuffer.extend(rews)
                # logger.record_tabular("EpLenMean", np.mean(lenbuffer))
                # logger.record_tabular("EpRewMean", np.mean(rewbuffer))
                # logger.record_tabular("EpThisIter", len(lens))
                episodes_so_far += len(lens)
                timesteps_so_far += sum(lens)
                iters_so_far += 1
                # logger.record_tabular("EpisodesSoFar", episodes_so_far)
                # logger.record_tabular("TimestepsSoFar", timesteps_so_far)
                # logger.record_tabular("TimeElapsed", time.time() - tstart)
                # if MPI.COMM_WORLD.Get_rank() == 0:
                #     logger.dump_tabular()

            # print(time.time() - tstart)
            self.push(self._get_policy())
            del env

        self.train_phase = train_phase
        # self.curpi = self.policy_fn("curpi%d" % self.pi_cnt, self.name, self.ob_space,
        #                             self.ac_space)  # Network for submitted policy
        # self.assign_cur_eq_new = U.function([], [], updates=[tf.assign(curv, newv)
        #                                                      for (curv, newv) in
        #                                                      zipsame(self.curpi.get_variables(),
        #                                                              self.pi.get_variables())])

    def _traj_segment_generator(self, pi, q_func, env, horizon, stochastic):
        t = 0
        ac = env.action_space.sample()  # not used, just so we have the datatype
        new = True  # marks if we're on first timestep of an episode
        ob = env.reset()

        cur_ep_ret = 0  # return in current episode
        cur_ep_len = 0  # len of current episode
        ep_rets = []  # returns of completed episodes in this segment
        ep_lens = []  # lengths of ...

        # Initialize history arrays
        obs = np.array([ob for _ in range(horizon)])
        rews = np.zeros(horizon, 'float32')
        qs = np.zeros(horizon, 'float32')
        news = np.zeros(horizon, 'int32')
        acs = np.array([ac for _ in range(horizon)])
        prevacs = acs.copy()

        while True:
            prevac = ac
            # ac, _ = pi.act(stochastic, ob)
            ac, _ = pi.act_with_explore(stochastic, ob, 1e-1)
            q = q_func(ob[None], ac[None])

            # Slight weirdness here because we need value function at time T
            # before returning segment [0, T-1] so we get the correct
            # terminal value
            if t > 0 and t % horizon == 0:
                # print ({"ob": obs, "rew": rews, "vpred": vpreds, "new": news,
                #        "ac": acs, "prevac": prevacs, "nextvpred": vpred * (1 - new),
                #        "ep_rets": ep_rets, "ep_lens": ep_lens})
                yield {"ob": obs, "rew": rews, "q": qs, "new": news,
                       "ac": acs, "prevac": prevacs, "nextq": q * (1 - new),
                       "ep_rets": ep_rets, "ep_lens": ep_lens}
                # Be careful!!! if you change the downstream algorithm to aggregate
                # several of these batches, then be sure to do a deepcopy
                ep_rets = []
                ep_lens = []
            i = t % horizon
            obs[i] = ob
            qs[i] = q
            news[i] = new
            acs[i] = ac
            prevacs[i] = prevac

            ob, rew, _, new = env.step(ac)
            # print(new)
            rews[i] = rew

            cur_ep_ret += rew
            cur_ep_len += 1
            if new:
                ep_rets.append(cur_ep_ret)
                ep_lens.append(cur_ep_len)
                cur_ep_ret = 0
                cur_ep_len = 0
                ob = env.reset()
            t += 1

    @staticmethod
    def _add_q_target(seg, gamma):
        """
        Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
        """
        new = np.append(seg["new"],
                        0)  # last element is only used for last vtarg, but we already zeroed it if last new = 1
        # vpred = np.append(seg["vpred"], seg["nextvpred"])
        q = np.append(seg["q"], seg["nextq"])
        T = len(seg["rew"])
        # print(T, seg["rew"])
        # print(T)
        # seg["adv"] = gaelam = np.empty(T, 'float32')
        # seg["delta"] = delta = np.empty(T, 'float32')
        rew = seg["rew"]
        seg["qtarget"] = qtarget = np.empty(T, "float32")
        # lastgaelam = 0
        for t in reversed(range(T)):
            nonterminal = 1 - new[t + 1]
            # print(nonterminal)
            qtarget[t] = rew[t] + gamma * q[t + 1] * nonterminal
        # print(seg["adv"])
        # print("adv", seg["adv"])
        # print("vpred", seg["vpred"])
        # print("seg", seg)

    def _get_policy(self):
        # self.assign_cur_eq_new()
        # with tf.variable_scope(self.scope):
            # curpi = self.policy_fn("curpi%d" % self.pi_cnt, self.ob_space, self.ac_space)  # Network for submitted policy
            # self.pi_cnt += 1
            # curpi = self.curpi
            # assign_cur_eq_new = U.function([], [], updates=[tf.assign(curv, newv)
            #                                                 for (curv, newv) in
            #                                                 zipsame(curpi.get_variables(), self.pi.get_variables())])
            # assign_cur_eq_new()
            # delete ass

        def act_fn(ob):
            # if type(self.exploration) == float:
            #     ac, _ = self.curpi.act_with_explore(stochastic=True, ob=ob, explore_prob=self.exploration)
            # elif self.exploration is None:
            #     ac, _ = self.curpi.act(stochastic=True, ob=ob)
            # else:
            #     raise NotImplementedError
            ac = self.pi.act(stochastic=True, ob=ob)
            return ac

        def prob_fn(ob, ac):
            return self.pi.prob(ob, ac)

        return Policy(act_fn, prob_fn)

    def get_initial_policy(self):
        return self._get_policy()

    def get_final_policy(self):
        return self._get_policy()

    def train(self, *args, **kwargs):
        self.train_phase(*args, **kwargs)

    def get_config(self):
        return {}






def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]