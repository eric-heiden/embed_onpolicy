import click
import imageio
import numpy as np
import os.path as osp
import pickle
import sys

from PIL import ImageFont, Image, ImageDraw

sys.path.insert(0, osp.join(osp.dirname(__file__), 'baselines'))
sys.path.insert(0, osp.join(osp.dirname(__file__), 'garage'))

from baselines import logger
from baselines.common import set_global_seeds
import baselines.her.experiment.config as config
from baselines.her.rollout import RolloutWorker

from run_push_her import prepare_params

from garage.envs.mujoco.sawyer import PushEnv

from collections import deque

import numpy as np
import pickle
from mujoco_py import MujocoException

from baselines.her.util import convert_episode_to_batch_major, store_args

TRAJ_SIZE = 30
orange = np.array([255, 163, 0])
red = np.array([255, 0, 0])
blue = np.array([20, 163, 255])
width_factor = 1. / TRAJ_SIZE * 512
lower_part = 512 // 5
max_reward = 1.
font = ImageFont.truetype("Consolas.ttf", 32)


class RenderWorker:

    @store_args
    def __init__(self, make_env, policy, dims, logger, T, rollout_batch_size=1,
                 exploit=False, use_target_net=False, compute_Q=False, noise_eps=0,
                 random_eps=0, history_len=100, render=False, **kwargs):
        """Rollout worker generates experience by interacting with one or many environments.

        Args:
            make_env (function): a factory function that creates a new instance of the environment
                when called
            policy (object): the policy that is used to act
            dims (dict of ints): the dimensions for observations (o), goals (g), and actions (u)
            logger (object): the logger that is used by the rollout worker
            rollout_batch_size (int): the number of parallel rollouts that should be used
            exploit (boolean): whether or not to exploit, i.e. to act optimally according to the
                current policy without any exploration
            use_target_net (boolean): whether or not to use the target net for rollouts
            compute_Q (boolean): whether or not to compute the Q values alongside the actions
            noise_eps (float): scale of the additive Gaussian noise
            random_eps (float): probability of selecting a completely random action
            history_len (int): length of history for statistics smoothing
            render (boolean): whether or not to render the rollouts
        """
        self.envs = [make_env() for _ in range(rollout_batch_size)]
        for env in self.envs:
            env.render()
        assert self.T > 0

        self.info_keys = [key.replace('info_', '') for key in dims.keys() if key.startswith('info_')]

        self.success_history = deque(maxlen=history_len)
        self.Q_history = deque(maxlen=history_len)

        self.n_episodes = 0
        self.g = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # goals
        self.initial_o = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)  # observations
        self.initial_ag = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # achieved goals
        self.reset_all_rollouts()
        self.clear_history()

        self.video = []

    def reset_rollout(self, i):
        """Resets the `i`-th rollout environment, re-samples a new goal, and updates the `initial_o`
        and `g` arrays accordingly.
        """
        obs = self.envs[i].reset()
        self.initial_o[i] = obs['observation']
        self.initial_ag[i] = obs['achieved_goal']
        self.g[i] = obs['desired_goal']

    def reset_all_rollouts(self):
        """Resets all `rollout_batch_size` rollout workers.
        """
        for i in range(self.rollout_batch_size):
            self.reset_rollout(i)

    def generate_rollouts(self):
        """Performs `rollout_batch_size` rollouts in parallel for time horizon `T` with the current
        policy acting on it accordingly.
        """
        self.reset_all_rollouts()

        # compute observations
        o = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)  # observations
        ag = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # achieved goals
        o[:] = self.initial_o
        ag[:] = self.initial_ag

        # generate episodes
        obs, achieved_goals, acts, goals, successes = [], [], [], [], []
        info_values = [np.empty((self.T, self.rollout_batch_size, self.dims['info_' + key]), np.float32) for key in self.info_keys]
        Qs = []
        for t in range(self.T):
            policy_output = self.policy.get_actions(
                o, ag, self.g,
                compute_Q=self.compute_Q,
                noise_eps=self.noise_eps if not self.exploit else 0.,
                random_eps=self.random_eps if not self.exploit else 0.,
                use_target_net=self.use_target_net)

            if self.compute_Q:
                u, Q = policy_output
                Qs.append(Q)
            else:
                u = policy_output

            if u.ndim == 1:
                # The non-batched case should still have a reasonable shape.
                u = u.reshape(1, -1)

            o_new = np.empty((self.rollout_batch_size, self.dims['o']))
            ag_new = np.empty((self.rollout_batch_size, self.dims['g']))
            success = np.zeros(self.rollout_batch_size)
            # compute new states and observations
            for i in range(self.rollout_batch_size):
                rewards = []
                infos = []
                try:
                    # We fully ignore the reward here because it will have to be re-computed
                    # for HER.
                    curr_o_new, reward, _, info = self.envs[i].step(u[i])
                    rewards.append(reward)
                    infos.append(info)
                    if 'is_success' in info:
                        success[i] = info['is_success']
                    o_new[i] = curr_o_new['observation']
                    ag_new[i] = curr_o_new['achieved_goal']
                    for idx, key in enumerate(self.info_keys):
                        info_values[idx][t, i] = info[key]

                    env = self.envs[i]  # type: PushEnv
                    env.render_camera = "camera_side"
                    img_left = env.render(mode='rgb_array')
                    env.render_camera = "camera_topdown"

                    img_center = env.render(mode='rgb_array')
                    # render rewards
                    img_center[-lower_part, :10] = orange
                    img_center[-lower_part, -10:] = orange
                    if TRAJ_SIZE < 512:
                        p_rew_x = 0
                        for j, r in enumerate(rewards):
                            rew_x = int(j * width_factor)
                            if r < 0:
                                color = blue if infos[j]["grasped"] else red
                                img_center[-1:, p_rew_x:rew_x] = color
                                img_center[-1:, p_rew_x:rew_x] = color
                            else:
                                rew_y = int(r / max_reward * lower_part)
                                color = blue if infos[j]["grasped"] else orange
                                img_center[-rew_y - 1:, p_rew_x:rew_x] = color
                                img_center[-rew_y - 1:, p_rew_x:rew_x] = color
                            p_rew_x = rew_x
                    else:
                        for j, r in enumerate(rewards):
                            rew_x = int(j * width_factor)
                            if r < 0:
                                color = blue if infos[j]["grasped"] else red
                                img_center[-1:, rew_x] = color
                                img_center[-1:, rew_x] = color
                            else:
                                rew_y = int(r / max_reward * lower_part)
                                color = blue if infos[j]["grasped"] else orange
                                img_center[-rew_y - 1:, rew_x] = color
                                img_center[-rew_y - 1:, rew_x] = color

                    env.render_camera = "camera_front"
                    img_right = env.render(mode='rgb_array')
                    img_left = Image.fromarray(np.uint8(img_left))
                    draw_left = ImageDraw.Draw(img_left)
                    draw_left.text((20, 20), "Batch %i" % (i + 1), fill="black", font=font)
                    img_right = Image.fromarray(np.uint8(img_right))
                    draw_right = ImageDraw.Draw(img_right)
                    draw_right.text((20, 20), "Step %i" % info["l"], fill="black", font=font)
                    self.video.append(np.hstack((np.array(img_left), np.array(img_center), np.array(img_right))))

                except MujocoException as e:
                    return self.generate_rollouts()

            if np.isnan(o_new).any():
                self.logger.warning('NaN caught during rollout generation. Trying again...')
                self.reset_all_rollouts()
                return self.generate_rollouts()

            obs.append(o.copy())
            achieved_goals.append(ag.copy())
            successes.append(success.copy())
            acts.append(u.copy())
            goals.append(self.g.copy())
            o[...] = o_new
            ag[...] = ag_new
        obs.append(o.copy())
        achieved_goals.append(ag.copy())
        self.initial_o[:] = o

        episode = dict(o=obs,
                       u=acts,
                       g=goals,
                       ag=achieved_goals)
        for key, value in zip(self.info_keys, info_values):
            episode['info_{}'.format(key)] = value

        # stats
        successful = np.array(successes)[-1, :]
        assert successful.shape == (self.rollout_batch_size,)
        success_rate = np.mean(successful)
        self.success_history.append(success_rate)
        if self.compute_Q:
            self.Q_history.append(np.mean(Qs))
        self.n_episodes += self.rollout_batch_size

        imageio.mimsave('play_her.mp4', self.video, fps=20)

        return convert_episode_to_batch_major(episode)

    def clear_history(self):
        """Clears all histories that are used for statistics
        """
        self.success_history.clear()
        self.Q_history.clear()

    def current_success_rate(self):
        return np.mean(self.success_history)

    def current_mean_Q(self):
        return np.mean(self.Q_history)

    def save_policy(self, path):
        """Pickles the current policy for later inspection.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.policy, f)

    def logs(self, prefix='worker'):
        """Generates a dictionary that contains all collected statistics.
        """
        logs = []
        logs += [('success_rate', np.mean(self.success_history))]
        if self.compute_Q:
            logs += [('mean_Q', np.mean(self.Q_history))]
        logs += [('episode', self.n_episodes)]

        if prefix is not '' and not prefix.endswith('/'):
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs

    def seed(self, seed):
        """Seeds each environment with a distinct seed derived from the passed in global seed.
        """
        for idx, env in enumerate(self.envs):
            env.seed(seed + 1000 * idx)



@click.command()
@click.argument('policy_file', type=str, default="/home/eric/embed_onpolicy/log/push_her_0_2018-08-13-13-12-42/policy_best.pkl")
@click.option('--seed', type=int, default=0)
@click.option('--n_test_rollouts', type=int, default=10)
@click.option('--render', type=int, default=1)
def main(policy_file, seed, n_test_rollouts, render):
    set_global_seeds(seed)

    # Load policy.
    with open(policy_file, 'rb') as f:
        policy = pickle.load(f)
    env_name = policy.info['env_name']

    # Prepare params.
    params = config.DEFAULT_PARAMS
    if env_name in config.DEFAULT_ENV_PARAMS:
        params.update(config.DEFAULT_ENV_PARAMS[env_name])  # merge env-specific parameters in
    params['env_name'] = env_name
    params = prepare_params(params)
    config.log_params(params, logger=logger)

    dims = config.configure_dims(params)

    eval_params = {
        'exploit': True,
        'use_target_net': params['test_with_polyak'],
        'compute_Q': True,
        'rollout_batch_size': 1,
        'render': bool(render),
    }

    for name in ['T', 'gamma', 'noise_eps', 'random_eps']:
        eval_params[name] = params[name]

    evaluator = RenderWorker(params['make_env'], policy, dims, logger, **eval_params)
    evaluator.seed(seed)

    # Run evaluation.
    evaluator.clear_history()
    for _ in range(n_test_rollouts):
        evaluator.generate_rollouts()

    # record logs
    for key, val in evaluator.logs('test'):
        logger.record_tabular(key, np.mean(val))
    logger.dump_tabular()


if __name__ == '__main__':
    main()
