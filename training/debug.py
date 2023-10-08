#! /usr/bin/env python
# import gym
# from absl import app, flags
# import rail_walker_gym
# import matplotlib.pyplot as plt
import os

import gym
import numpy as np
import tqdm
import wandb
from absl import app, flags
from flax.training import checkpoints
from jaxrl5.agents import SACLearner
from jaxrl5.agents.sac.sac_learner_wdynamics import SACLearnerWithDynamics
from jaxrl5.agents.agent import Agent as JaxRLAgent
from jaxrl5.data import ReplayBuffer
from jaxrl5.wrappers import wrap_gym
from ml_collections import config_flags
import pickle
from dm_control import viewer
from dm_control.mjcf import export_with_assets
import typing

import mujoco
from natsort import natsorted 
import shutil
import rail_walker_gym
import rail_walker_interface

from checkpoint_util import initialize_project_log, load_latest_checkpoint, save_checkpoint, load_latest_replay_buffer, save_replay_buffer, save_rollout
from eval_util import evaluate
from task_config_util import apply_task_configs

import matplotlib.pyplot as plt

FLAGS = flags.FLAGS
flags.DEFINE_string('env_name', 'Go1SanityMujoco-v0', 'Environment name.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_boolean('generate_data', False, 'Generate data.')

def import_register():
    if "Real" in FLAGS.env_name:
        import rail_walker_gym.envs.register_real
    if "Mujoco" in FLAGS.env_name:
        import rail_walker_gym.envs.register_mujoco

def main(_):
    import_register()
    env : gym.Env = gym.make(FLAGS.env_name)
    env = wrap_gym(env, rescale_actions=True)
    env.seed(FLAGS.seed)
    
    # import ipdb; ipdb.set_trace()
    # export_with_assets(
    #     env.task.root_entity.mjcf_model,
    #     out_dir = "./kevin",
    #     out_file_name="scene.xml"
    # )
    # env = gym.wrappers.RescaleAction(env, -1, 1)
    # env = gym.wrappers.ClipAction(env)
    # env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
    # import ipdb; ipdb.set_trace()
    if FLAGS.generate_data:
        actions = []
    else:
        actions = np.load("actions.npy")
    returns = []
    
    env.joystick_policy.robot.foot_friction = 0.2
    for i in range(10):
        traj_actions = []
        curr_return = 0
        env.reset()
        for j in range(100):
            if FLAGS.generate_data:
                action = env.action_space.sample()
            else:
                action = actions[i][j]
            obs, reward, done, info = env.step(action)
            if FLAGS.generate_data:
                traj_actions.append(action)
            curr_return += reward

        returns.append(curr_return)
        if FLAGS.generate_data:
            actions.append(traj_actions)
        print("Return:", curr_return)

    if FLAGS.generate_data:
        np.save("returns.npy", np.array(returns))
        np.save("actions.npy", np.array(actions))

    # save a video of the saved images
    # import imageio
    # images = []
    # for i in range(50):
        # images.append(imageio.imread(f"img_{i}.png"))
    # imageio.mimsave('movie.gif', images, fps=2)

    # delete images
    # import os
    # for i in range(50):
        # os.remove(f"img_{i}.png")

    # from ipdb import set_trace; set_trace()


if __name__ == '__main__':
    app.run(main)
