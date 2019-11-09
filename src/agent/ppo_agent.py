import time
from collections import deque

from monitor.statistics import Statistics
from baselines.common import zipsame
from agent.base_agent import BaseAgent
from agent.policy import Policy
import baselines.common.tf_util as U
from baselines.common import Dataset
import tensorflow as tf, numpy as np
from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_moments import mpi_moments
from mpi4py import MPI
import json
from common.path_utils import *


class PPOAgent(BaseAgent):
    def __init__(self, name, *args, **kwargs):
        self.name = name
        with tf.variable_scope(name):
            # print(name)
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name
            # print(self.scope)

    def get_config(self):
        return self.config

    def save(self, save_path):
        json.dump(self.config, open(join_path_and_check(save_path, "config.json"), "w"))
        U.save_variables(join_path(save_path, "model.obj"),
                         variables=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name))

    def load(self, load_path):
        U.load_variables(join_path(load_path, "model.obj"),
                         variables=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name))

    def _init(self, policy_fn, ob_space, ac_space,
              handlers,
              timesteps_per_actorbatch,
              clip_param, entcoeff,  # clipping parameter epsilon, entropy coeff
              optim_epochs, optim_stepsize,  # optimization hypers
              gamma, lam,  # advantage estimation
              max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
              callback=None,  # you can do anything in the callback, since it takes locals(), globals()
              adam_epsilon=1e-5,
              beta1=0.9,
              schedule='constant',  # annealing for stepsize parameters (epsilon and adam)
              opponent='latest',  # opponent type
              exploration=None
              ):
        self.config = {
            "timesteps_per_actorbatch": timesteps_per_actorbatch,
            "clip_param": clip_param,
            "entcoeff": entcoeff,
            "optim_epochs": optim_epochs,
            "optim_stepsize": optim_stepsize,
            "gamma": gamma,
            "lam": lam,
            "max_timesteps": max_timesteps,
            "max_episodes": max_episodes,
            "max_iters": max_iters,
            "max_seconds": max_seconds,
            "adam_epsilon": adam_epsilon,
            "schedule": schedule,
            "opponent": opponent
        }
        # print(ob_space)
        self.schedule = schedule
        self.policy_fn = policy_fn
        self.ob_space = ob_space
        self.ac_space = ac_space
        self.push, self.pull = handlers
        self.exploration = exploration
        pi = policy_fn("pi", self.name, ob_space, ac_space)  # Construct network for new policy
        oldpi = policy_fn("oldpi", self.name, ob_space, ac_space)  # Network for old policy
        self.avgpi = avgpi = policy_fn("avgpi", self.name, ob_space, ac_space)
        atarg = tf.placeholder(dtype=tf.float32, shape=[None])  # Target advantage function (if applicable)
        prob = tf.placeholder(dtype=tf.float32, shape=[None])  # Visiting probability
        ret = tf.placeholder(dtype=tf.float32, shape=[None])  # Empirical return

        lrmult = tf.placeholder(name='lrmult', dtype=tf.float32,
                                shape=[])  # learning rate multiplier, updated with schedule

        ob = U.get_placeholder_cached(name="ob_" + self.name)
        ac = pi.pdtype.sample_placeholder([None])

        kloldnew = oldpi.pd.kl(pi.pd)
        ent = pi.pd.entropy()
        meankl = tf.reduce_mean(kloldnew)
        meanent = tf.reduce_mean(ent)
        pol_entpen = (-entcoeff) * meanent

        ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac))  # pnew / pold
        new_targ = atarg# / prob
        surr1 = ratio * new_targ  # surrogate from conservative policy iteration
        surr2 = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * new_targ  #
        pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2))  # PPO's pessimistic surrogate (L^CLIP)
        # pol_surr = tf.reduce_mean(pi.pd.logp(ac) * atarg)
        vf_loss = tf.reduce_mean(tf.square(pi.vpred - ret))
        total_loss = pol_surr + pol_entpen + vf_loss
        losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
        loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

        var_list = pi.get_trainable_variables()
        lossandgrad = U.function([ob, ac, atarg, prob, ret, lrmult], losses + [U.flatgrad(total_loss, var_list)])
        adam = MpiAdam(var_list, epsilon=adam_epsilon, beta1=beta1)
        # adam =

        assign_old_eq_new = U.function([], [], updates=[tf.assign(oldv, newv)
                                                        for (oldv, newv) in
                                                        zipsame(oldpi.get_variables(), pi.get_variables())])
        compute_losses = U.function([ob, ac, atarg, prob, ret, lrmult], losses)

        avg_alpha = tf.placeholder(dtype=tf.float32, shape=[])
        update_avg = U.function([avg_alpha], [], updates=[tf.assign(avgv, tf.multiply(avgv, (1 - avg_alpha)) +
                                                                    tf.multiply(newv, avg_alpha))
                                                          for (avgv, newv) in
                                                          zipsame(avgpi.get_trainable_variables(),
                                                                  pi.get_trainable_variables())])

        U.initialize()
        adam.sync()

        self.cnt = 1
        update_avg(1. / self.cnt)

        self.gamma = gamma
        self.lam = lam
        self.pi = pi
        self.pi_cnt = 0

        self.average_utility = 0.0
        self.tot = 0

        def train_phase(i, statistics: Statistics, progress, ep_i):
            # print(self.scope + ("avg util: %.5f" % self.average_utility))
            env = self.pull(opponent)
            seg_gen = self._traj_segment_generator(self.pi, env, timesteps_per_actorbatch, stochastic=True)

            episodes_so_far = 0
            timesteps_so_far = 0
            iters_so_far = 0
            tstart = time.time()
            lenbuffer = deque(maxlen=100)  # rolling buffer for episode lengths
            rewbuffer = deque(maxlen=100)  # rolling buffer for episode rewards

            assert sum([max_iters > 0, max_timesteps > 0, max_episodes > 0,
                        max_seconds > 0]) == 1, "Only one time constraint permitted"

            rews = []
            vpreds = []
            while True:
                if callback: callback(locals(), globals())
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
                self._add_vtarg_and_adv(seg, gamma, lam)

                rews.append(np.average(seg["rew"]))
                vpreds.append(np.average(seg["vpred"]))

                # if i == 1:
                #     print(seg["vpred"])

                # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
                # logger.log(seg["rew"])
                ob, ac, atarg, tdlamret, prob = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"], seg["prob"]
                # print(prob)
                # logger.log(atarg)
                vpredbefore = seg["vpred"]  # predicted value function before udpate
                # if atarg.std() > 1e-4:
                #     atarg = (atarg - atarg.mean()) / atarg.std()  # standardized advantage function estimate
                # else:
                #     print(atarg.mean())
                #     atarg = (atarg - atarg.mean())
                    # print(atarg)
                d = Dataset(dict(ob=ob, ac=ac, atarg=atarg, vtarg=tdlamret, prob=prob), deterministic=pi.recurrent)
                optim_batchsize = ob.shape[0]
                # print("optim_batchsize:", optim_batchsize)

                if type(self.schedule) == tuple:
                    schedule, k = self.schedule
                else:
                    schedule = self.schedule
                if schedule == 'constant':
                    cur_lrmult = 1.0
                elif schedule == "dec":
                    cur_lrmult = np.exp(-progress)
                elif schedule == 'linear':
                    cur_lrmult = max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
                elif schedule == "wolf":
                    if np.average(seg["rew"]) < self.average_utility:
                        cur_lrmult = k
                    else:
                        cur_lrmult = 1.0
                elif schedule == "wolf2":
                    if np.average(seg["rew"]) < self.average_utility - 0.01:
                        cur_lrmult = k
                    else:
                        cur_lrmult = 1.0
                    # cur_lrmult *= max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
                elif schedule == "wolf_adv":
                    if np.average(seg["delta"]) > 0.:
                        cur_lrmult = 1.0
                    else:
                        cur_lrmult = k
                elif schedule == "wolf_adv2":
                    if np.average(seg["adv"]) > 0.:
                        cur_lrmult = 1.0
                    else:
                        cur_lrmult = k
                elif schedule == "cfr":
                    if np.average(seg["adv"]) > 0.:
                        cur_lrmult = 1.
                    else:
                        cur_lrmult = 1.
                elif schedule == "wolf_stat":
                    assert len(seg["rew"]) == 1
                    if seg["rew"][0] > statistics.get_avg_rew(i, seg["ob"][0]):
                        cur_lrmult = 1.0
                    else:
                        cur_lrmult = k
                elif schedule == "wolf_stat_matrix":
                    if np.average(seg["rew"]) > statistics.get_avg_rew(i, seg["ob"][0]):
                        cur_lrmult = 1.0
                    else:
                        cur_lrmult = k
                elif schedule == "sqrt":
                    cur_lrmult = 1.0 / np.sqrt(ep_i + 1)
                else:
                    raise NotImplementedError

                # cur_lrmult *= 1e-2 / np.sqrt(progress + 1e-2)
                # print(1e-2 / np.sqrt(progress))

                self.average_utility = self.average_utility * self.tot + np.sum(seg["rew"])
                self.tot += len(seg["rew"])
                self.average_utility /= self.tot

                # fasdaqwe = 0.9
                # self.average_utility = fasdaqwe * self.average_utility + np.average(seg["rew"]) * (1 - fasdaqwe)

                # print(self.scope + ("avg util: %.5f" % self.average_utility))

                if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob)  # update running mean/std for policy

                assign_old_eq_new()  # set old parameter values to new parameter values
                # logger.log("Optimizing...")
                # logger.log(fmt_row(13, loss_names))
                # Here we do a bunch of optimization epochs over the data
                for _ in range(optim_epochs):
                    losses = []  # list of tuples, each of which gives the loss for a minibatch
                    for batch in d.iterate_once(optim_batchsize):
                        # print(batch["prob"])
                        *newlosses, g = lossandgrad(batch["ob"], batch["ac"], batch["atarg"], batch["prob"],
                                                    batch["vtarg"], cur_lrmult)
                        adam.update(g, optim_stepsize * cur_lrmult)
                    #     losses.append(newlosses)
                    # logger.log(fmt_row(13, np.mean(losses, axis=0)))

                # logger.log("Evaluating losses...")
                losses = []
                for batch in d.iterate_once(optim_batchsize):
                    newlosses = compute_losses(batch["ob"], batch["ac"], batch["atarg"], batch["prob"], batch["vtarg"],
                                               cur_lrmult)
                    # print(newlosses)
                    losses.append(newlosses)
                meanlosses, _, _ = mpi_moments(losses, axis=0)
                # logger.log(fmt_row(13, meanlosses))
                # for (lossval, name) in zipsame(meanlosses, loss_names):
                #     logger.record_tabular("loss_" + name, lossval)
                # logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))
                lrlocal = (seg["ep_lens"], seg["ep_rets"])  # local values
                listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal)  # list of tuples
                lens, rews = map(flatten_lists, zip(*listoflrpairs))
                lenbuffer.extend(lens)
                rewbuffer.extend(rews)
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
            self.cnt += 1
            update_avg(1. / self.cnt)
            self.assign_cur_eq_new()
            self.push(self._get_policy(self.curpi), self._get_policy(self.avgpi))
            del env
            return {
                "rews": rews,
                "vpreds": vpreds
            }

        self.train_phase = train_phase
        self.curpi = self.policy_fn("curpi%d" % self.pi_cnt, self.name, self.ob_space, self.ac_space)  # Network for submitted policy
        self.assign_cur_eq_new = U.function([], [], updates=[tf.assign(curv, newv)
                                                            for (curv, newv) in
                                                            zipsame(self.curpi.get_variables(), self.pi.get_variables())])
        self.assign_cur_eq_new()

    def _traj_segment_generator(self, pi, env, horizon, stochastic):
        t = 0
        ac = env.action_space.sample()  # not used, just so we have the datatype
        new = True  # marks if we're on first timestep of an episode
        ob, prob, _ = env.reset()
        # print(ob, prob)

        cur_ep_ret = 0  # return in current episode
        cur_ep_len = 0  # len of current episode
        ep_rets = []  # returns of completed episodes in this segment
        ep_lens = []  # lengths of ...

        # Initialize history arrays
        obs = np.array([ob for _ in range(horizon)])
        rews = np.zeros(horizon, 'float32')
        vpreds = np.zeros(horizon, 'float32')
        news = np.zeros(horizon, 'int32')
        acs = np.array([ac for _ in range(horizon)])
        prevacs = acs.copy()
        probs = np.zeros(horizon, 'float32')

        while True:
            prevac = ac
            if type(self.exploration) == float:
                ac, vpred = pi.act_with_explore(stochastic, ob, self.exploration)
            elif self.exploration is None:
                # ac, vpred = pi.act_with_explore(stochastic, ob, .1)
                # print(ob)
                ac, vpred = pi.act(stochastic, ob)
            else:
                raise NotImplementedError

            my_prob = pi.prob(ob, ac)

            # Slight weirdness here because we need value function at time T
            # before returning segment [0, T-1] so we get the correct
            # terminal value
            if t > 0 and t % horizon == 0:
                # print ({"ob": obs, "rew": rews, "vpred": vpreds, "new": news,
                #        "ac": acs, "prevac": prevacs, "nextvpred": vpred * (1 - new),
                #        "ep_rets": ep_rets, "ep_lens": ep_lens})
                # print(vpreds)
                yield {"ob": obs, "rew": rews, "vpred": vpreds, "new": news,
                       "ac": acs, "prevac": prevacs, "nextvpred": vpred * (1 - new),
                       "ep_rets": ep_rets, "ep_lens": ep_lens, "prob": probs}
                # Be careful!!! if you change the downstream algorithm to aggregate
                # several of these batches, then be sure to do a deepcopy
                ep_rets = []
                ep_lens = []
            i = t % horizon
            obs[i] = ob
            vpreds[i] = vpred
            news[i] = new
            acs[i] = ac
            prevacs[i] = prevac
            probs[i] = prob

            ob, rew, _, new, prob, _ = env.step(ac, my_prob)
            # print("1", ob, prob)
            # print(new)
            rews[i] = rew

            cur_ep_ret += rew
            cur_ep_len += 1
            if new:
                ep_rets.append(cur_ep_ret)
                ep_lens.append(cur_ep_len)
                cur_ep_ret = 0
                cur_ep_len = 0
                ob, prob, _ = env.reset()
            t += 1

    @staticmethod
    def _add_vtarg_and_adv(seg, gamma, lam):
        """
        Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
        """
        new = np.append(seg["new"],
                        0)  # last element is only used for last vtarg, but we already zeroed it if last new = 1
        vpred = np.append(seg["vpred"], seg["nextvpred"])
        T = len(seg["rew"])
        # print(T, seg["rew"])
        # print(T)
        seg["adv"] = gaelam = np.empty(T, 'float32')
        seg["delta"] = delta = np.empty(T, 'float32')
        rew = seg["rew"]
        lastgaelam = 0
        for t in reversed(range(T)):
            nonterminal = 1 - new[t + 1]
            # print(nonterminal)
            delta[t] = rew[t] + gamma * vpred[t + 1] * nonterminal - vpred[t]
            # print(delta)
            # delta[t] = rew[t]
            gaelam[t] = lastgaelam = delta[t] + gamma * lam * nonterminal * lastgaelam
        # print(seg["adv"])
        seg["tdlamret"] = seg["adv"] + seg["vpred"]
        # print("adv", seg["adv"])
        # print("vpred", seg["vpred"])
        # print("seg", seg)

    def _get_policy(self, pi):
        def act_fn(ob):
            if type(self.exploration) == float:
                ac, _ = pi.act_with_explore(stochastic=True, ob=ob, explore_prob=self.exploration)
            elif self.exploration is None:
                ac, _ = pi.act(stochastic=True, ob=ob)
            else:
                raise NotImplementedError
            return ac

        def prob_fn(ob, ac):
            return pi.prob(ob, ac)

        def strategy_fn(ob):
            return pi.strategy(ob)

        return Policy(act_fn, prob_fn, strategy_fn)

    def get_initial_policy(self):
        return self._get_policy(self.curpi)

    def get_final_policy(self):
        return self._get_policy(self.avgpi)

    def get_average_policy(self):
        return self._get_policy(self.avgpi)

    def train(self, *args, **kwargs):
        return self.train_phase(*args, **kwargs)


def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]