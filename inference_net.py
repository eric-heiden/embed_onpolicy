import numpy as np
import tensorflow as tf
from gym.spaces import Discrete, Box
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm
from baselines.common.distributions import make_pdtype
from baselines import bench, logger

from policies import space_input


class InferenceNetwork(object):
    def __init__(self, sess: tf.Session, ob_space: Box, ac_space: Box, latent_space: Box, horizon: int,
                 nbatch, learning_rate=1e-3):

        with tf.variable_scope("inference_network"):
            # observation input
            Observation, processed_ob = space_input(ob_space, nbatch, name="ob")
            processed_ob = tf.layers.flatten(processed_ob, name="flattened_ob")

            # action input
            Action, processed_ac = space_input(ac_space, nbatch, name="ac")
            processed_ac = tf.layers.flatten(processed_ac, name="flattened_ac")

            ob_ac = tf.concat((processed_ob, processed_ac), axis=1, name="ob_ac")
            ob_ac.apply(tf.contrib.data.sliding_window_batch(window_size=horizon, stride=1))

            with tf.name_scope("embedding"):
                em_h1 = tf.tanh(fc(ob_ac, 'embed_fc1', nh=8, init_scale=0.5), name="em_h1")
                em_h2 = tf.tanh(fc(em_h1, 'embed_fc2', nh=latent_space.shape[0], init_scale=0.5), name="em_h2")
                em_logstd = tf.get_variable(name='em_logstd', shape=[1, ac_space.shape[0]],
                                         initializer=tf.zeros_initializer(), trainable=True)
                em_std = tf.exp(em_logstd, name="em_std")
                self.em_pd = tf.distributions.Normal(em_h2, em_std, name="embedding", validate_args=True)

            # latent input
            Embedding, processed_em = space_input(latent_space, nbatch, name="em")
            processed_em = tf.layers.flatten(processed_em, name="flattened_em")
            processed_em.apply(tf.contrib.data.sliding_window_batch(window_size=horizon, stride=1))

            loss = self.em_pd.log_prob(Embedding, "log_likelihood")
            params = tf.trainable_variables()
            print("TRAINABLE VARS", params)

            with tf.name_scope("Training"):
                grads = tf.gradients(loss, params)
                grads = list(zip(grads, params))
                trainer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-5, name="adam_opt")
                _train = trainer.apply_gradients(grads)

        self.Observation = Observation
        self.Action = Action
        self.Embedding = Embedding
