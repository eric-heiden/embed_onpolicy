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
@click.argument('config_file', type=str,
                default="/home/eric/.deep-rl-docker/embed_onpolicy/log/push_pos_embed_1234_2018-08-18-16-06-59/configuration.pkl")
@click.option('--checkpoint', type=str, default="00450")
@click.option('--interactive', type=bool, default=True)
@click.option('--n_test_rollouts', type=int, default=30)
@click.option('--n_cherrypick_trials', type=int, default=1)
def main(config_file, checkpoint, interactive, n_test_rollouts, n_cherrypick_trials):
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
    unwrap_env = configuration["unwrap_env_fn"]
    envs = [configuration["make_env"](task=task) for task in range(ntasks)]
    use_embedding = "use_embedding" not in configuration or configuration["use_embedding"]

    # initialize GL
    envs[0].render()

    sampler = Sampler(env=envs[0], unwrap_env=unwrap_env, model=model, traj_size=configuration["traj_size"],
                      inference_opt_epochs=1,
                      inference_coef=lambda _: 0,
                      gamma=configuration["gamma"], lam=configuration["lambda"],
                      use_embedding=use_embedding)

    rollout_dir = os.path.join(config_path, "playbacks")
    if not os.path.exists(rollout_dir):
        os.makedirs(rollout_dir)

    for round in range(n_test_rollouts):
        print('####### Sampling round %i #######' % (round + 1))
        for task, env in enumerate(envs):
            if interactive:
                obs, returns, masks, actions, values, neglogpacs, latents, tasks, states, epinfos, \
                completions, inference_loss, inference_log_likelihoods, inference_discounted_log_likelihoods, \
                extras = sampler.run(0, env, task, render=None, interactive=True)
            else:
                sampled_video = None
                epinfos = None
                for cherry in range(max(1, n_cherrypick_trials)):
                    if n_cherrypick_trials > 1:
                        print("Sampling task %i (cherrypick trial %i of %i)..." % (
                        task + 1, cherry + 1, n_cherrypick_trials))

                    rf = configuration["render_fn"](task, iteration)
                    obs, returns, masks, actions, values, neglogpacs, latents, tasks, states, epinfos, \
                    completions, inference_loss, inference_log_likelihoods, inference_discounted_log_likelihoods, \
                    sampled_video, extras = sampler.run(0, env, task, render=rf)
                    if any([info["d"] for info in epinfos]):
                        print('SUCCESS')
                        break

                if sampled_video is not None:
                    imageio.mimsave(osp.join(rollout_dir, 'embed_%05d_task%02d_%02d.mp4' % (iteration, task, round)),
                                    sampled_video, fps=60)

                if epinfos is not None:
                    if "joints" not in epinfos[0]:
                        print("No joint information available.")
                    else:
                        joints = np.array([epinfo["joints"] for epinfo in epinfos])
                        unwrapped = unwrap_env(env)
                        qpos = {
                            joint_name: joints[:, joint_id] for joint_id, joint_name in
                        enumerate(unwrapped.sim.model.joint_names)
                        }
                        with open(osp.join(rollout_dir, 'embed_%05d_task%02d_%02d.pkl' % (iteration, task, round)),
                                  'wb') as fh:
                            fh.write(cloudpickle.dumps(qpos))
            print("...completed %i / %i steps" % (len(epinfos), configuration["traj_size"]))


if __name__ == '__main__':
    main()
