import click
import cloudpickle
import imageio
import numpy as np
import os
import os.path as osp
import sys
import tensorflow as tf

sys.path.insert(0, osp.join(osp.dirname(__file__), 'baselines'))
sys.path.insert(0, osp.join(osp.dirname(__file__), 'garage'))

from model import Model
from sampler import Sampler

@click.command()
@click.argument('config_file', type=str, default="/home/eric/embed_onpolicy/log/reacher2_embed_12345_2018-08-08-19-55-30/configuration.pkl")
@click.option('--checkpoint', type=str, default="latest")
@click.option('--n_test_rollouts', type=int, default=10)
def main(config_file, checkpoint, n_test_rollouts):
    configuration = cloudpickle.load(open(config_file, "rb"))
    # {
    #     "make_model": make_model,
    #     "make_env": env_fn,
    #     "render_fn": render_fn,
    #     "traj_plot_fn": traj_plot_fn,
    #     "unwrap_env_fn": unwrap_env,
    #     "traj_size": traj_size,
    #     "task_space": task_space,
    #     "latent_space": latent_space,
    #     "curriculum_fn": curriculum_fn,
    #     "seed": seed,
    #     "gamma": gamma,
    #     "lambda": lam,
    #     "vf_coef": vf_coef,
    #     "policy_entropy": policy_entropy,
    #     "embedding_entropy": embedding_entropy,
    #     "inference_horizon": inference_horizon,
    #     "max_grad_norm": max_grad_norm,
    #     "nbatches": nbatches,
    #     "total_timesteps": total_timesteps,
    #     "cliprange": cliprange,
    #     "lr": lr,
    #     "plot_folder": plot_folder
    # }
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
    checkpoint = os.path.join(checkpoints, checkpoint)
    print("Loading weights from checkpoint %s." % checkpoint)
    model.load(checkpoint)

    task_space = configuration["task_space"]
    ntasks = task_space.shape[0]
    unwrap_env = configuration["unwrap_env_fn"]
    envs = [configuration["make_env"](task=task) for task in range(ntasks)]

    # initialize GL
    envs[0].render()

    sampler = Sampler(env=envs[0], unwrap_env=unwrap_env, model=model, traj_size=configuration["traj_size"],
                      inference_opt_epochs=1,
                      inference_coef=0,
                      gamma=configuration["gamma"], lam=configuration["lambda"])

    rollout_dir = os.path.join(config_path, "playbacks")
    if not os.path.exists(rollout_dir):
        os.makedirs(rollout_dir)

    update = configuration["total_timesteps"] // configuration["traj_size"]
    for round in range(n_test_rollouts):
        print('####### Sampling round %i #######' % (round+1))
        for task, env in enumerate(envs):
            print("Sampling task %i..." % (task+1))

            rf = configuration["render_fn"](task, update)
            obs, returns, masks, actions, values, neglogpacs, latents, tasks, states, epinfos, \
            completions, inference_loss, inference_log_likelihoods, inference_discounted_log_likelihoods, \
            inference_means, inference_stds, sampled_video = sampler.run(env, task, render=rf)

            imageio.mimsave(osp.join(rollout_dir, 'embed_%05d_task%02d_%02d.mp4' % (update, task, round)), sampled_video, fps=20)

            joints = np.array([epinfo["joints"] for epinfo in epinfos])
            unwrapped = unwrap_env(env)
            qpos = {
                joint_name: joints[:, joint_id] for joint_id, joint_name in enumerate(unwrapped.sim.model.joint_names)
            }
            with open(osp.join(rollout_dir, 'embed_%05d_task%02d_%02d.pkl' % (update, task, round)), 'wb') as fh:
                fh.write(cloudpickle.dumps(qpos))
            print("...completed %i / %i steps" % (len(epinfos), configuration["traj_size"]))


if __name__ == '__main__':
    main()
