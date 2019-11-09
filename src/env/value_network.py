import tensorflow as tf
import baselines.common.tf_util as U
import os


def trim_name(name):
    print(name)
    return '/'.join(name.split('/')[1:])


def save_variables(save_path, variables=None, sess=None):
    import joblib
    sess = sess or U.get_session()
    variables = variables or tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    ps = sess.run(variables)
    save_dict = {trim_name(v.name): value for v, value in zip(variables, ps)}
    dirname = os.path.dirname(save_path)
    if any(dirname):
        os.makedirs(dirname, exist_ok=True)
    joblib.dump(save_dict, save_path)


def load_variables(load_path, variables=None, sess=None):
    import joblib
    sess = sess or U.get_session()
    variables = variables or tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    loaded_params = joblib.load(os.path.expanduser(load_path))
    restores = []
    if isinstance(loaded_params, list):
        assert len(loaded_params) == len(variables), 'number of variables loaded mismatches len(variables)'
        for d, v in zip(loaded_params, variables):
            restores.append(v.assign(d))
    else:
        for v in variables:
            restores.append(v.assign(loaded_params[trim_name(v.name)]))

    sess.run(restores)


class ValueNetwork(object):
    def __init__(self, name, input_length, width, depth):
        with tf.variable_scope(name):
            self.name = name
            self.scope = tf.get_variable_scope().name
            ob = tf.placeholder(dtype=tf.float32, shape=[None, input_length], name='ob')
            last_out = ob
            for i in range(depth):
                last_out = tf.nn.relu(tf.layers.dense(last_out, width, name="fc%i" % (i + 1),
                                                      kernel_initializer=U.normc_initializer(1.0)))
            last_out = tf.layers.dense(last_out, 1, name='final', kernel_initializer=U.normc_initializer(1.0))[:, 0]
            self._calc = U.function([ob], last_out)

            v = tf.placeholder(dtype=tf.float32, shape=[None], name='v')
            loss = tf.reduce_mean(tf.square(last_out - v))
            adam = tf.compat.v1.train.AdamOptimizer()
            self.loss_step = U.function([ob, v], adam.minimize(loss, var_list=self.get_trainable_variables()))
            U.initialize()

    def calc(self, ob):
        return self._calc(ob[None])[0]

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def save(self, save_path, name=None):
        if name is None:
            name = self.name
        save_variables(os.path.join(save_path, "{}.model".format(name)), self.get_trainable_variables())

    def load(self, load_path, name=None):
        if name is None:
            name = self.name
        load_variables(os.path.join(load_path, "{}.model".format(name)), self.get_trainable_variables())
