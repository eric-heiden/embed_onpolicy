import numpy as np
import tensorflow as tf
from gym.spaces import Discrete, Box
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm
from baselines.common.distributions import make_pdtype
from baselines import bench, logger


class InferenceNetwork(object):
    def __init__(self, sess: tf.Session, ob_space: Box, ac_space: Box, latent_space: Box, horizon: int,
                 learning_rate=1e-3, gamma=0.99):
        print("Initializing inference network with H = %d" % horizon)
        self.horizon = horizon
        with tf.variable_scope("inference_network"):
            input_size = (ob_space.shape[0] + ac_space.shape[0]) * horizon
            latent_size = latent_space.shape[0]

            with tf.name_scope("traj_window"):
                # ob_ac = tf.concat((Observation, Action), axis=1, name="ob_ac")
                traj_window = tf.placeholder(shape=(None, input_size),
                                       dtype=tf.float32,
                                       name="traj_window")
                # shape = sess.run([ob_ac.get_shape()])
                # print(shape)
                # ob_ac = tf.data.Dataset.from_tensors((processed_ob, processed_ac))
                # ob_ac = ob_ac.apply(tf.contrib.data.sliding_window_batch(window_size=horizon, stride=1))

            with tf.name_scope("embedding"):
                em_h1 = tf.tanh(fc(traj_window, 'embed_fc1', nh=16, init_scale=0.5), name="em_h1")
                em_h2 = tf.tanh(fc(em_h1, 'embed_fc2', nh=latent_size, init_scale=0.5), name="em_h2")
                em_logstd = tf.get_variable(name='em_logstd', shape=[1, latent_size],
                                            initializer=tf.zeros_initializer(), trainable=True)
                em_std = tf.exp(em_logstd, name="em_std")
                self.em_pd = tf.distributions.Normal(em_h2, em_std, name="embedding", validate_args=True)

            with tf.name_scope("true_latent"):
                # latent input
                # Embedding = horizon_space_input(latent_space, horizon, None, name="em")
                Embedding = tf.placeholder(shape=(None, latent_size),
                                           dtype=tf.float32,
                                           name="em")
                # processed_em = tf.layers.flatten(processed_em, name="flattened_em")
                # processed_em.apply(tf.contrib.data.sliding_window_batch(window_size=horizon, stride=1))

            with tf.name_scope("loss"):
                ll = tf.reduce_sum(self.em_pd.log_prob(Embedding), name="log_likelihood")
                discount = tf.placeholder(dtype=tf.float32, shape=[None], name="discounts")
                discounted_ll = discount * ll

                loss = tf.identity(-tf.reduce_mean(discounted_ll), name="traj_enc_loss")
                params = tf.trainable_variables()
                print("TRAINABLE VARS", params)

            with tf.name_scope("training"):
                grads = tf.gradients(loss, params)
                grads = list(zip(grads, params))
                trainer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-5, name="adam_opt")
                _train = trainer.apply_gradients(grads)

            def train(traj_windows, discounts, latents):
                return sess.run([loss, discounted_ll, _train], {
                    traj_window: traj_windows,
                    discount: discounts,
                    Embedding: latents
                })[:-1]

        # self.Observation = Observation
        # self.Action = Action
        # self.Embedding = Embedding
        self.train = train
