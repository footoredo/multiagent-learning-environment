from baselines.common.mpi_running_mean_std import RunningMeanStd
import baselines.common.tf_util as U
import tensorflow as tf
import gym
from baselines.common.distributions import make_pdtype
import numpy as np


class MLPPolicy(object):
    recurrent = False

    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            # print(name)
            # print(tf.get_variable_scope().name)
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, agent_name, ob_space, ac_space, hid_size, num_hid_layers, gaussian_fixed_var=True):
        assert isinstance(ob_space, gym.spaces.Box)

        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None

        # print(ob_space.shape)
        ob = U.get_placeholder(name="ob_" + agent_name, dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))

        with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape=ob_space.shape)

        with tf.variable_scope('vf'):
            # obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
            obz = ob
            last_out = obz
            for i in range(num_hid_layers):
                last_out = tf.nn.tanh(tf.layers.dense(last_out, hid_size, name="fc%i"%(i+1), kernel_initializer=U.normc_initializer(1.0)))
            self.vpred = tf.layers.dense(last_out, 1, name='final', kernel_initializer=U.normc_initializer(1.0))[:,0]
            self._vp = U.function([ob], self.vpred)

        with tf.variable_scope('pol'):
            last_out = obz
            for i in range(num_hid_layers):
                last_out = tf.nn.tanh(tf.layers.dense(last_out, hid_size, name='fc%i'%(i+1), kernel_initializer=U.normc_initializer(1.0)))
            if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
                mean = tf.layers.dense(last_out, pdtype.param_shape()[0]//2, name='final', kernel_initializer=U.normc_initializer(0.01))
                logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0]//2], initializer=tf.zeros_initializer())
                pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
            else:
                pdparam = tf.layers.dense(last_out, pdtype.param_shape()[0], name='final', kernel_initializer=U.normc_initializer(0.01))

        self.pd = pdtype.pdfromflat(pdparam)

        self.state_in = []
        self.state_out = []

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        self._act = U.function([stochastic, ob], [ac, self.vpred])
        # acq = tf.placeholder(dtype=tf.int32, shape=())
        # prob = tf.nn.softmax(self.pd.logits)[acq]
        # print(self.pd.logits)
        # print(self.pd.mean)
        prob = self.pd.mean
        self._prob = U.function([ob], prob)

    def act(self, stochastic, ob):
        # print(ob)
        # print(self.scope, ob)
        # print(ob, ob[None])
        ac1, vpred1 = self._act(stochastic, ob[None])
        # print(ac1, vpred1)
        return ac1[0], vpred1[0]

    def prob(self, ob, ac):
        # print(ob)
        prob1 = self._prob(ob[None])
        return prob1[0][ac]

    def vp(self, ob):
        return self._vp(ob[None])[0]

    def strategy(self, ob):
        # print(ob, ob[None], self._prob(ob[None])[0])
        return self._prob(ob[None])[0]

    def act_with_explore(self, stochastic, ob, explore_prob):
        if np.random.rand() < explore_prob:
            return np.array(np.random.choice(2)), 0.
        else:
            ac1, vpred1 = self._act(stochastic, ob[None])
            return ac1[0], vpred1[0]

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_initial_state(self):
        return []
