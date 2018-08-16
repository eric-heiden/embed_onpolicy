import numpy as np
import tensorflow as tf
from gym.spaces import Box
from baselines.a2c.utils import fc


class InferenceNetwork(object):
    def __init__(self, sess: tf.Session, ob_space: Box, ac_space: Box, latent_space: Box, horizon: int,
                 learning_rate=1e-3, hidden_layers=(16,)):
        print("Initializing inference network with H = %d" % horizon)
        self.horizon = horizon
        with tf.variable_scope("inference_network"):
            input_size = (ob_space.shape[0] + ac_space.shape[0]) * horizon
            latent_size = latent_space.shape[0]

            with tf.name_scope("traj_window"):
                self.traj_window = tf.placeholder(shape=(None, input_size),
                                       dtype=tf.float32,
                                       name="traj_window")

            with tf.name_scope("embedding"):
                layer_in = self.traj_window
                for i, units in enumerate(hidden_layers):
                    em_h = tf.tanh(fc(layer_in, 'embed_fc%i' % (i+1), nh=units, init_scale=0.5), name="em_h%i" % (i+1))
                    layer_in = em_h
                em_h = tf.tanh(fc(layer_in, 'embed_fc', nh=latent_size, init_scale=0.5), name="em_h")
                em_logstd = tf.get_variable(name='em_logstd', shape=[1, latent_size],
                                            initializer=tf.zeros_initializer(), trainable=True)
                em_std = tf.exp(em_logstd, name="em_std")
                self.em_pd = tf.distributions.Normal(em_h, em_std, name="embedding", validate_args=False)
                self.embedding_mean = self.em_pd.mean("embedding_mean")
                self.embedding_std = self.em_pd.stddev("embedding_std")

            with tf.name_scope("true_latent"):
                # latent input
                Embedding = tf.placeholder(shape=(None, latent_size),
                                           dtype=tf.float32,
                                           name="em")
                # processed_em = tf.layers.flatten(processed_em, name="flattened_em")

            with tf.name_scope("loss"):
                ll = tf.reduce_sum(self.em_pd.log_prob(Embedding), axis=1, name="log_likelihood")
                # ll = self.em_pd.log_prob(Embedding, name="log_likelihood")
                discount = tf.placeholder(dtype=tf.float32, shape=[None], name="discounts")
                discounted_ll = discount * ll

                loss = tf.identity(-tf.reduce_mean(discounted_ll), name="traj_enc_loss")
                params = tf.trainable_variables()
                print("TRAINABLE INFERENCENET VARS", params)

            with tf.name_scope("training"):
                grads = tf.gradients(loss, params)
                grads = list(zip(grads, params))
                trainer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-5, name="adam_opt")
                _train = trainer.apply_gradients(grads)

            def train(traj_windows, discounts, latents):
                return sess.run([loss, ll, discounted_ll, _train], {
                    self.traj_window: traj_windows,
                    discount: discounts,
                    Embedding: latents
                })[:-1]

            def embedding_params(traj_window):
                return sess.run([self.embedding_mean, self.embedding_std], {self.traj_window: traj_window})

        self.train = train
        self.embedding_params = embedding_params
