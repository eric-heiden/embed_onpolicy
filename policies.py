import numpy as np
import tensorflow as tf
from gym.spaces import Discrete, Box
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm
from baselines.common.distributions import make_pdtype

EPS = 1e-7


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
                 traj_size, reuse=False, name="model", use_beta=False, seed=None,
                 em_hidden_layers=(8,), pi_hidden_layers=(16, 16), vf_hidden_layers=(16, 16),
                 activation_fn=tf.nn.leaky_relu, embedding_actiation_fn=tf.identity):

        with tf.variable_scope(name, reuse=reuse):
            # task input
            with tf.name_scope("task"):
                Task, processed_t = space_input(task_space, 1, name="task")
                processed_t = tf.layers.flatten(processed_t, "flattened_t")

            # observation input
            Observation, processed_ob = space_input(ob_space, None, name="ob")
            processed_ob = tf.layers.flatten(processed_ob, name="flattened_ob")

            # embedding network (with task as input)
            with tf.name_scope("embedding"):
                layer_in = processed_t
                for i, units in enumerate(em_hidden_layers):
                    em_h = embedding_actiation_fn(fc(layer_in, 'embed_fc%i' % (i+1), nh=units, init_scale=0.5), name="em_h%i" % (i+1))
                    layer_in = em_h
                em_h = embedding_actiation_fn(fc(layer_in, 'embed_fc', nh=latent_space.shape[0], init_scale=0.5), name="em_h")
                em_logstd = tf.get_variable(name='em_logstd', shape=[1, latent_space.shape[0]],
                                            initializer=tf.zeros_initializer(), trainable=True)
                em_std = tf.exp(em_logstd, name="em_std")
                self.em_pd = tf.distributions.Normal(em_h, em_std, name="embedding", validate_args=False)

                self.embedding_mean = self.em_pd.mean("embedding_mean")
                self.embedding_std = self.em_pd.stddev("embedding_std")

                # embedding variable
                Embedding = self.em_pd.sample(name="em", seed=seed)
                self.tiled_em = tf.tile(Embedding, (traj_size, 1), name="tiled_embedding")[:tf.shape(Observation)[0]]

            with tf.name_scope("embedding_entropy"):
                embedding_entropy = tf.reduce_mean(self.em_pd.entropy(), name="embedding_entropy")

            # embedding + observation input
            em_ob = tf.concat((self.tiled_em, processed_ob), axis=1, name="em_ob")

            # policy
            pi_input = em_ob
            with tf.name_scope("pi"):
                for i, units in enumerate(pi_hidden_layers):
                    pi_h = activation_fn(fc(pi_input, 'pi_fc%i' % (i+1), nh=units, init_scale=np.sqrt(2)), name="pi_h%i" % (i+1))
                    pi_input = pi_h

            self.inference_lll = tf.placeholder(dtype=tf.float32, shape=[None], name="inference_lll")

            # value function
            with tf.name_scope("vf"):
                tiled_t = tf.tile(Task, (traj_size, 1), name="tiled_task")[:tf.shape(Observation)[0]]
                # vf_input = tf.concat((processed_ob, tiled_t), axis=1, name="ob_task")
                vf_input = tf.concat((self.tiled_em, processed_ob, tiled_t), axis=1, name="em_ob_task")
                # vf_input = tf.concat((self.tiled_em, processed_ob, tiled_t, self.inference_lll), axis=1, name="em_ob_task_inf")
                for i, units in enumerate(vf_hidden_layers):
                    vf_h = activation_fn(fc(vf_input, 'vf_fc%i' % (i+1), nh=units, init_scale=np.sqrt(2)), name="vf_h%i" % (i+1))
                    vf_input = vf_h
                vf = fc(vf_input, 'vf', 1)[:, 0]

            if use_beta:
                # use Beta distribution
                with tf.name_scope("PolicyDist_beta"):
                    alpha = tf.nn.softplus(fc(pi_input, 'pi_alpha1', ac_space.shape[0], init_scale=1., init_bias=0.), name='pi_alpha')
                    beta = tf.nn.softplus(fc(pi_input, 'pi_beta1', ac_space.shape[0], init_scale=1., init_bias=0.), name='pi_beta')
                    clipped_alpha = tf.clip_by_value(alpha, clip_value_min=EPS, clip_value_max=-np.log(EPS))
                    clipped_beta = tf.clip_by_value(beta, clip_value_min=EPS, clip_value_max=-np.log(EPS))
                    self.pd = tf.distributions.Beta(clipped_alpha, clipped_beta, validate_args=False, name="PolicyDist_beta")
            else:
                # use Gaussian distribution
                with tf.name_scope("PolicyDist_normal"):
                    mean = fc(pi_input, 'pi', ac_space.shape[0], init_scale=0.01, init_bias=0.)
                    logstd = tf.get_variable(name='pi_logstd', shape=[1, ac_space.shape[0]],
                                             initializer=tf.zeros_initializer(), trainable=True)
                    std = tf.exp(logstd)
                    self.pd = tf.distributions.Normal(mean, std, allow_nan_stats=False, name="PolicyDist_normal")

            with tf.name_scope("action"):
                # TODO revert
                # action = self.pd.mean(name="action")
                action = self.pd.sample(name="action", seed=seed)
                action_mean = self.pd.mean("action_mean")
                action_mode = self.pd.mode("action_mode")

            l, h = ac_space.low, ac_space.high
            if use_beta:
                with tf.name_scope("transform_action"):
                    action = action * (h - l) + l
                    action_mean = action_mean * (h - l) + l
                    action_mode = action_mode * (h - l) + l

            def neg_log_prob(var: tf.Tensor, var_name="var"):
                with tf.name_scope("neg_log_prob_%s" % var_name):
                    if use_beta:
                        var = (var - l) / (h-l)
                        return tf.identity(-tf.reduce_sum(self.pd.log_prob(tf.clip_by_value(var, EPS, 1. - EPS)), axis=-1), name="neg_log_prob_%s" % var_name)
                    else:
                        return tf.identity(-tf.reduce_sum(self.pd.log_prob(var), axis=-1), name="neg_log_prob_%s" % var_name)

            neglogp0 = neg_log_prob(action, "action")
            self.initial_state = None

        def step(latent, ob, task, *_args, action_type="sample", **_kwargs):
            # XXX task is only fed for the value function!
            if action_type == "mean":
                a, v, neglogp = sess.run([action_mean, vf, neglogp0], {Observation: ob, Embedding: latent, Task: task})
            elif action_type == "mode":
                a, v, neglogp = sess.run([action_mode, vf, neglogp0], {Observation: ob, Embedding: latent, Task: task})
            else:
                a, v, neglogp = sess.run([action, vf, neglogp0], {Observation: ob, Embedding: latent, Task: task})

            # print("Observation len: ", l, " true:", np.shape(ob))
            return a, v, self.initial_state, neglogp

        def step_from_task(task, ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([action, vf, neglogp0], {Observation: ob, Task: task})
            return a, v, self.initial_state, neglogp

        def value(latent, ob, task, *_args, **_kwargs):
            return sess.run(vf, {Observation: ob, Embedding: latent, Task: task})

        def value_from_task(task, ob, *_args, **_kwargs):
            return sess.run(vf, {Observation: ob, Task: task})

        def latent_from_task(task):
            return sess.run(Embedding, {Task: task})

        def embedding_params(task):
            return sess.run([self.embedding_mean, self.embedding_std], {Task: task})

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
        self.embedding_entropy = embedding_entropy
        self.embedding_params = embedding_params
