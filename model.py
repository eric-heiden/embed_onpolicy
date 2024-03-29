from typing import Callable

import tensorflow as tf
import numpy as np
import joblib

from inference_net import InferenceNetwork
from policies import MlpEmbedPolicy

TimeVarying = Callable[[int], float]


def const_fn(val):
    return lambda _: val


class Model(object):
    def __init__(self, *, policy, ob_space, ac_space, task_space, latent_space, traj_size, cliprange: TimeVarying,
                 lr: TimeVarying, policy_entropy: TimeVarying, vf_coef: TimeVarying, max_grad_norm,
                 embedding_entropy: TimeVarying = const_fn(0.), inference_horizon=5, seed=None, em_hidden_layers=(8,),
                 pi_hidden_layers=(16, 16), vf_hidden_layers=(16, 16), inference_hidden_layers=(16,),
                 embed_grad_factor=1e4,
                 use_embedding=True, **_kwargs):

        self.traj_size = traj_size
        self.policy_entropy = policy_entropy
        self.embedding_entropy = embedding_entropy
        self.inference_horizon = inference_horizon
        self.vf_coef = vf_coef
        self.seed = seed
        self.cliprange = cliprange
        self.lr = lr

        self.parameters = {
            "traj_size": traj_size,
            "policy_entropy": policy_entropy(0),
            "embedding_entropy": embedding_entropy(0),
            "inference_horizon": inference_horizon,
            "vf_coef": vf_coef(0),
            "lr": lr(0),
            "cliprange": cliprange(0),
            "seed": ('None' if seed is None else seed)
        }

        sess = tf.get_default_session()
        with tf.variable_scope("PPO"):
            act_model = policy(sess, ob_space, ac_space, task_space, latent_space, traj_size=1,
                               reuse=False, seed=seed, name="model",
                               em_hidden_layers=em_hidden_layers,
                               pi_hidden_layers=pi_hidden_layers,
                               vf_hidden_layers=vf_hidden_layers,
                               use_embedding=use_embedding)  # type: MlpEmbedPolicy
            train_model = policy(sess, ob_space, ac_space, task_space, latent_space, traj_size=traj_size,
                                 reuse=True, seed=seed, name="model",
                                 em_hidden_layers=em_hidden_layers,
                                 pi_hidden_layers=pi_hidden_layers,
                                 vf_hidden_layers=vf_hidden_layers,
                                 use_embedding=use_embedding)  # type: MlpEmbedPolicy

            A = tf.placeholder(dtype=tf.float32, shape=train_model.pd.batch_shape, name="actions")
            # A = train_model.pd.sample(name="A")
            ADV = tf.placeholder(tf.float32, [None], name="advantages")
            # ADV2 = tf.stack((ADV, ADV), axis=1, name="stacked_ADV")
            R = tf.placeholder(tf.float32, [None], name="returns")
            OLDNEGLOGPAC = tf.placeholder(tf.float32, [None], name="old_neglogpac")
            OLDVPRED = tf.placeholder(tf.float32, [None], name="old_vpred")

            # time-varying parameters
            LR = tf.placeholder(tf.float32, [], name="learning_rate")
            CLIPRANGE = tf.placeholder(tf.float32, [], name="clip_range")
            VF_COEFF = tf.placeholder(tf.float32, [], name="vf_coeff")
            PI_ENT_COEF = tf.placeholder(tf.float32, [], name="pi_ent_coef")
            EM_ENT_COEF = tf.placeholder(tf.float32, [], name="em_ent_coef")

            neglogpac = train_model.neg_log_prob(A, "neglogpac")
            entropy = tf.nn.softplus(tf.reduce_mean(train_model.pd.entropy(), name="entropy"))

            with tf.name_scope("ValueFunction"):
                vpred = train_model.vf
                vpredclipped = tf.identity(OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, -CLIPRANGE, CLIPRANGE,
                                                                       name="clip_vf"), name="vpred_clipped")
                vf_losses1 = tf.square(vpred - R, name="vf_loss1")
                vf_losses2 = tf.square(vpredclipped - R, name="vf_loss2")
                vf_loss = tf.identity(.5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2)), name="vf_loss")

            with tf.name_scope("PolicyGradient"):
                ratio = tf.exp(OLDNEGLOGPAC - neglogpac, name="nlp_ratio")
                pg_losses = tf.identity(-ADV * ratio, name="pg_loss1")
                pg_losses2 = tf.identity(-ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE),
                                         name="pg_loss2")
                pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2), name="pg_loss")
                approxkl = tf.identity(.5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC)), name="approx_kl")
                clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)), name="clip_frac")
                loss = tf.identity(pg_loss + vf_loss * VF_COEFF, name="policy_loss")

                if use_embedding:
                    final_loss = tf.identity(loss
                                             - PI_ENT_COEF * entropy
                                             - EM_ENT_COEF * train_model.embedding_entropy,
                                             name="final_loss")
                else:
                    final_loss = tf.identity(loss - PI_ENT_COEF * entropy,
                                             name="final_loss")

            with tf.variable_scope('model', reuse=True):
                params = tf.trainable_variables(scope="PPO")
                print("TRAINABLE VARS", params)

            with tf.name_scope("Training"):
                src_grads = tf.gradients(final_loss, params)
                if max_grad_norm is not None:
                    src_grads, _grad_norm = tf.clip_by_global_norm(src_grads, max_grad_norm, name="clipped_grads")
                src_grads = tuple(src_grads)
                # raw_grads = tf.convert_to_tensor(src_grads)
                # tf.assign(src_grads, src_grads)
                # src_grads[0] = raw_grads
                grads = list(zip(src_grads, params))
                trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5, name="adam_opt")
                # grads = trainer.compute_gradients(final_loss, var_list=params)
                _train = trainer.apply_gradients(grads)

        def train(iteration: int, batches):
            self.parameters["policy_entropy"] = policy_entropy(iteration)
            self.parameters["embedding_entropy"] = embedding_entropy(iteration)
            self.parameters["vf_coef"] = vf_coef(iteration)
            self.parameters["lr"] = lr(iteration)
            self.parameters["cliprange"] = cliprange(iteration)

            computed_latents = []
            gradients = None
            losses = []
            advantages = np.array(
                [returns - values for obs, tasks, returns, masks, actions, values, neglogpacs, states in
                 batches]).flatten()

            # compute batch-wise gradients / losses
            for obs, tasks, returns, masks, actions, values, neglogpacs, states in batches:
                advs = returns - values
                advs = (advs - advs.mean()) / (advs.std() + 1e-8)
                td_map = {
                    train_model.Observation: obs,
                    train_model.Task: [tasks[0]],
                    A: actions,
                    ADV: advs,
                    R: returns,
                    CLIPRANGE: self.cliprange(iteration),
                    VF_COEFF: self.vf_coef(iteration),
                    PI_ENT_COEF: self.policy_entropy(iteration),
                    EM_ENT_COEF: self.embedding_entropy(iteration),
                    OLDNEGLOGPAC: neglogpacs,
                    OLDVPRED: values
                }
                if states is not None:
                    td_map[train_model.S] = states
                    td_map[train_model.M] = masks

                if use_embedding:
                    result = sess.run(
                        [pg_loss, vf_loss, approxkl, clipfrac, entropy, train_model.embedding_entropy, final_loss,
                         train_model.tiled_em, src_grads],
                        td_map
                    )
                else:
                    result = sess.run(
                        [pg_loss, vf_loss, approxkl, clipfrac, entropy, final_loss, src_grads],
                        td_map
                    )

                losses.append(result[:len(self.loss_names)])
                computed_latents.append(result[-2])
                if gradients is None:
                    gradients = [[result[-1][i]] for i in range(len(result[-1]))]
                else:
                    for i in range(len(result[-1])):
                        gradients[i].append(result[-1][i])

            # update weights via batch-normalized gradients
            normalized_grads = []
            for i in range(len(gradients)):
                normalized_grads.append(np.mean(gradients[i], axis=0))

            # normalized_grads[:5] *= np.full(5, embed_grad_factor)

            sess.run([_train], {
                src_grads: normalized_grads,
                LR: lr(iteration),
                CLIPRANGE: cliprange(iteration)
            })

            if use_embedding:
                grad_means = {name: np.mean(g) for g, name in zip(normalized_grads[:5],
                                                                  map(str, params[:5]))}
                print("Embedding grads:", grad_means)

            return np.mean(losses, axis=0), computed_latents, advantages

        def get_latent(task: int):
            if not use_embedding:
                return None
            one_hot = np.zeros(act_model.Task.shape)
            one_hot[:, task] = 1
            latent = sess.run(act_model.Embedding, {act_model.Task: one_hot})
            return latent[0]

        if use_embedding:
            self.loss_names = ['policy_loss', 'value_loss', 'approxkl', 'clipfrac', 'policy_entropy',
                               'embedding_entropy', 'final_loss']
        else:
            self.loss_names = ['policy_loss', 'value_loss', 'approxkl', 'clipfrac', 'policy_entropy',
                               'final_loss']

        def save(save_path):
            ps = sess.run(params)
            joblib.dump(ps, save_path)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            sess.run(restores)
            # If you want to load weights, also save/load observation scaling inside VecNormalize

        self.train = train
        self.train_model = train_model

        if use_embedding:
            self.inference_model = InferenceNetwork(sess, ob_space, ac_space, latent_space,
                                                    horizon=inference_horizon, hidden_layers=inference_hidden_layers)
        self.act_model = act_model
        self.step = act_model.step
        self.step_from_task = act_model.step_from_task
        self.value = act_model.value
        self.value_from_task = act_model.value_from_task
        self.get_latent = get_latent
        self.initial_state = act_model.initial_state
        self.save = save
        self.load = load
        self.task_space = task_space
        self.latent_space = latent_space
        self.action_space = ac_space
        self.observation_space = ob_space

        self.use_beta = act_model.use_beta
        tf.global_variables_initializer().run(session=sess)
