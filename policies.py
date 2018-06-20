import numpy as np
import tensorflow as tf
from gym.spaces import Discrete, Box
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm
from baselines.common.distributions import make_pdtype
from baselines import bench, logger


def space_input(space, batch_size=None, name='Ob'):
    """
    Build gym.spaces input with encoding depending on the
    space type
    Params:

    space: space (should be one of gym.spaces)
    batch_size: batch size for input (default is None, so that resulting input placeholder can take tensors with any batch size)
    name: tensorflow variable name for input placeholder

    returns: tuple (input_placeholder, processed_input_tensor)
    """
    if isinstance(space, Discrete):
        input_x = tf.placeholder(shape=(batch_size,), dtype=tf.int32, name=name)
        processed_x = tf.to_float(tf.one_hot(input_x, space.n))
        return input_x, processed_x

    elif isinstance(space, Box):
        input_shape = (batch_size,) + space.shape
        input_x = tf.placeholder(shape=input_shape, dtype=space.dtype, name=name)
        processed_x = tf.to_float(input_x)
        return input_x, processed_x

    else:
        raise NotImplementedError


class MlpPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False):
        self.pdtype = make_pdtype(ac_space)
        with tf.variable_scope("model", reuse=reuse):
            X, processed_x = space_input(ob_space, nbatch)
            processed_x = tf.layers.flatten(processed_x)
            pi_h1 = tf.tanh(fc(processed_x, 'pi_fc1', nh=16, init_scale=np.sqrt(2)))
            pi_h2 = tf.tanh(fc(pi_h1, 'pi_fc2', nh=16, init_scale=np.sqrt(2)))
            vf_h1 = tf.tanh(fc(processed_x, 'vf_fc1', nh=16, init_scale=np.sqrt(2)))
            vf_h2 = tf.tanh(fc(vf_h1, 'vf_fc2', nh=16, init_scale=np.sqrt(2)))
            vf = fc(vf_h2, 'vf', 1)[:, 0]

            self.pd, self.pi = self.pdtype.pdfromlatent(pi_h2, init_scale=0.01)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X: ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X: ob})

        self.X = X
        self.vf = vf
        self.step = step
        self.value = value


class MlpEmbedPolicy(object):
    def __init__(self, sess: tf.Session, ob_space: Box, ac_space: Box, task_space: Box, latent_space: Box,
                 nbatch, nsteps, reuse=False, name="model", use_beta=False):

        # self.pdtype = make_pdtype(ac_space)
        with tf.variable_scope(name, reuse=reuse):

            # task input
            Task, processed_t = space_input(task_space, nbatch, name="task")
            processed_t = tf.layers.flatten(processed_t, "flattened_t")

            # embedding network (with task as input)
            em_h1 = tf.tanh(fc(processed_t, 'embed_fc1', nh=8, init_scale=0.5), name="em_h1")
            em_h2 = tf.tanh(fc(em_h1, 'embed_fc2', nh=latent_space.shape[0], init_scale=0.5), name="em_h2")
            em_h3 = tf.tanh(fc(em_h2, 'embed_fc3', nh=latent_space.shape[0], init_scale=0.5), name="em_h3")
            self.em_pd = tf.distributions.Normal(em_h2, em_h3, name="embedding")

            # embedding variable
            Embedding = self.em_pd.sample(name="em")  # tf.Variable(tf.zeros((nbatch,) + latent_space.shape), dtype=latent_space.dtype, name="embedding")
            # tf.assign(Embedding, em_h2, name="embedding_from_task")

            # observation input
            Observation, processed_ob = space_input(ob_space, nbatch, name="ob")
            processed_ob = tf.layers.flatten(processed_ob, name="flattened_ob")

            # embedding + observation input
            em_ob = tf.concat((Embedding, processed_ob), axis=1, name="em_ob")

            # policy
            pi_h1 = tf.tanh(fc(em_ob, 'pi_fc1', nh=16, init_scale=np.sqrt(2)), name="pi_h1")
            pi_h2 = tf.tanh(fc(pi_h1, 'pi_fc2', nh=16, init_scale=1.), name="pi_h2")
            # pi_h2 = tf.clip_by_value(pi_h2, ac_space.low[0], ac_space.high[0])

            # value function
            vf_h1 = tf.tanh(fc(em_ob, 'vf_fc1', nh=16, init_scale=np.sqrt(2)), name="vf_h1")
            vf_h2 = tf.tanh(fc(vf_h1, 'vf_fc2', nh=16, init_scale=np.sqrt(2)), name="vf_h2")
            vf = fc(vf_h2, 'vf', 1)[:, 0]

            if use_beta:
                # use Beta distribution
                alpha = tf.nn.softplus(fc(pi_h2, 'pi_alpha1', ac_space.shape[0], init_scale=1., init_bias=1.), name='pi_alpha')
                beta = tf.nn.softplus(fc(pi_h2, 'pi_beta1', ac_space.shape[0], init_scale=1., init_bias=1.), name='pi_beta')
                self.pd = tf.distributions.Beta(alpha + 0.0000001, beta + 0.0000001, validate_args=True, name="Beta")
            else:
                # use Gaussian distribution
                mean = fc(pi_h2, 'pi', ac_space.shape[0], init_scale=0.01, init_bias=0.)
                logstd = tf.get_variable(name='logstd', shape=[1, ac_space.shape[0]],
                                         initializer=tf.zeros_initializer(), trainable=True)
                std = tf.exp(logstd)
                self.pd = tf.distributions.Normal(mean, std, allow_nan_stats=False)

            a0 = self.pd.sample(name="a0")
            # a0 = self.pd.mean(name="a0")

            if use_beta:
                l, h = ac_space.low[0], ac_space.high[0]
                a0 = a0 * (h - l) + l

            def neg_log_prob(var: tf.Tensor, var_name="var"):
                if use_beta:
                    var = (var - l) / (h-l)
                    return -tf.reduce_sum(self.pd.log_prob(tf.clip_by_value(var, 0.00001, 0.999999)), axis=-1, name="neg_log_prob_%s" % var_name)
                else:
                    return -tf.reduce_sum(self.pd.log_prob(var), axis=-1, name="neg_log_prob_%s" % var_name)

            neglogp0 = neg_log_prob(a0, "a0")
            self.initial_state = None

        def step(latent, ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {Observation: ob, Embedding: latent})
            return a, v, self.initial_state, neglogp

        def step_from_task(task, ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {Observation: ob, Task: task})
            return a, v, self.initial_state, neglogp

        def value(latent, ob, *_args, **_kwargs):
            return sess.run(vf, {Observation: ob, Embedding: latent})

        def value_from_task(task, ob, *_args, **_kwargs):
            return sess.run(vf, {Observation: ob, Task: task})

        def latent_from_task(task):
            return sess.run(Embedding, {Task: task})

        self.Observation = Observation
        self.Task = Task
        self.Embedding = Embedding

        self.use_beta = use_beta

        self.vf = vf
        self.step = step
        self.step_from_task = step_from_task
        self.value = value
        self.value_from_task = value_from_task
        self.neg_log_prob = neg_log_prob
        self.latent_from_task = latent_from_task
