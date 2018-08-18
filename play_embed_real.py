import click
import cloudpickle
import imageio
import numpy as np
import os
import os.path as osp
import sys
import tensorflow as tf

import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import sys

import joblib
import moveit_commander
import numpy as np
import rospy
import tensorflow as tf

sys.path.insert(0, osp.join(osp.dirname(__file__), 'baselines'))
sys.path.insert(0, osp.join(osp.dirname(__file__), 'garage'))

from garage.contrib.ros.envs.sawyer import PusherEnv
from garage.misc import tensor_utils
from garage.tf.envs import TfEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv

from model import Model
from sampler import Sampler
from garage.envs.mujoco.sawyer.sawyer_env import SawyerEnv

@click.command()
@click.argument('config_file', type=str, default="/home/eric/.deep-rl-docker/embed_onpolicy/log/push_pos_embed_1234_2018-08-17-10-14-19/configuration.pkl")
@click.option('--checkpoint', type=str, default="latest")
@click.option('--task_id', type=int, default=0)
def main(config_file, checkpoint, task_id):
    configuration = cloudpickle.load(open(config_file, "rb"))
    print("Loaded configuration from %s." % config_file)

    ncpu = 1
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    tf.Session(config=config).__enter__()
    model = configuration["make_model"]()  # type: Model
    config_path = os.path.dirname(config_file)
    checkpoints = os.path.join(config_path, "checkpoints")
    if checkpoint == "latest":
        checkpoint = sorted(list(os.walk(checkpoints))[0][2])[-1]
    iteration = int(checkpoint)
    checkpoint = os.path.join(checkpoints, checkpoint)
    print("Loading weights from checkpoint %s." % checkpoint)
    model.load(checkpoint)

    task_space = configuration["task_space"]
    ntasks = task_space.shape[0]
    assert task_id in range(ntasks)
    unwrap_env = configuration["unwrap_env_fn"]
    env = configuration["make_env"](task=task_id)
    use_embedding = "use_embedding" not in configuration or configuration["use_embedding"]

    # initialize GL
    # env.render()

    sawyer_env = unwrap_env(env)  # type: SawyerEnv
    assert isinstance(sawyer_env, SawyerEnv)
    assert sawyer_env._control_method == "position_control"
    assert sawyer_env._start_configuration.joint_pos is not None
    assert sawyer_env._goal_configuration.object_pos is not None

    print("Initial joint angles:", sawyer_env._start_configuration.joint_pos)
    print("Object goal position:", sawyer_env._goal_configuration.object_pos)

    for key, value in configuration.items():
        if isinstance(value, int) or isinstance(value, str) or isinstance(value, float):
            print('%s:\t%s' % (key, value))

    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('rollingout_policy_push', anonymous=True)

    joint_config = {
        'right_j%i' % i: pos for i, pos in enumerate(sawyer_env._start_configuration.joint_pos)
    }



    push_env = PusherEnv(
        initial_goal=sawyer_env._goal_configuration.object_pos,
        initial_joint_pos=joint_config,
        simulated=False,
        robot_control_mode='position',
        action_scale=0.04
    )
    push_env._robot.set_joint_position_speed(0.05)
    rospy.on_shutdown(push_env.shutdown)


    push_env.initialize()

    vec_push_env = DummyVecEnv([lambda: push_env])

    def unwrap_env(env: DummyVecEnv, id: int = 0):
        return env.envs[id].env

    sampler = Sampler(env=vec_push_env, unwrap_env=unwrap_env, model=model, traj_size=configuration["traj_size"],
                      inference_opt_epochs=1,
                      inference_coef=0,
                      gamma=configuration["gamma"], lam=configuration["lambda"],
                      use_embedding=use_embedding)

    rollout_dir = os.path.join(config_path, "playbacks")
    if not os.path.exists(rollout_dir):
        os.makedirs(rollout_dir)
    obs, returns, masks, actions, values, neglogpacs, latents, tasks, states, epinfos, \
    completions, inference_loss, inference_log_likelihoods, inference_discounted_log_likelihoods, \
    extras = sampler.run(vec_push_env, task_id)
    if any([info["d"] for info in epinfos]):
        print('SUCCESS')

    print("Rollout completed after %i steps. Good bye!", len(obs))


if __name__ == '__main__':
    main()
