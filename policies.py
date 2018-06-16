import numpy as np
import tensorflow as tf
from gym.spaces import Discrete, Box
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm
from baselines.common.distributions import make_pdtype


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
                 nbatch, nsteps, reuse=False):

        self.pdtype = make_pdtype(ac_space)
        with tf.variable_scope("model", reuse=reuse):

            # task input
            Task, processed_t = space_input(task_space, nbatch, name="task")
            processed_t = tf.layers.flatten(processed_t)

            # embedding network (with task as input)
            em_h1 = tf.tanh(fc(processed_t, 'embed_fc1', nh=16, init_scale=np.sqrt(2)))
            em_h2 = tf.tanh(fc(em_h1, 'embed_fc2', nh=latent_space.shape[0], init_scale=np.sqrt(2)))

            # embedding variable
            Embedding = em_h2  # tf.Variable(tf.zeros((nbatch,) + latent_space.shape), dtype=latent_space.dtype, name="embedding")
            # tf.assign(Embedding, em_h2, name="embedding_from_task")

            # observation input
            Observation, processed_ob = space_input(ob_space, nbatch, name="ob")
            processed_ob = tf.layers.flatten(processed_ob, name="flattened_ob")

            # embedding + observation input
            em_ob = tf.concat((Embedding, processed_ob), axis=1, name="em_ob")

            # policy
            pi_h1 = tf.tanh(fc(em_ob, 'pi_fc1', nh=16, init_scale=np.sqrt(2)))
            pi_h2 = tf.tanh(fc(pi_h1, 'pi_fc2', nh=16, init_scale=np.sqrt(2)))

            # value function
            vf_h1 = tf.tanh(fc(em_ob, 'vf_fc1', nh=16, init_scale=np.sqrt(2)))
            vf_h2 = tf.tanh(fc(vf_h1, 'vf_fc2', nh=16, init_scale=np.sqrt(2)))
            vf = fc(vf_h2, 'vf', 1)[:, 0]

            self.pd, self.pi = self.pdtype.pdfromlatent(pi_h2, init_scale=0.01)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(latent, ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {Observation: ob, Embedding: latent})
            return a, v, self.initial_state, neglogp

        def value(latent, ob, *_args, **_kwargs):
            return sess.run(vf, {Observation: ob, Embedding: latent})

        self.Observation = Observation
        self.Task = Task
        self.Embedding = Embedding

        self.vf = vf
        self.step = step
        self.value = value
