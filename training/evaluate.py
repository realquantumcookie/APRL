#! /usr/bin/env python
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
import typing

import mujoco
from natsort import natsorted 
import shutil
import rail_walker_gym
import rail_walker_interface

from checkpoint_util import initialize_project_log, load_latest_checkpoint, save_rollout
from eval_util import evaluate
from task_config_util import apply_task_configs

import time

FLAGS = flags.FLAGS

# ==================== Training Flags ====================
flags.DEFINE_string('env_name', 'Go1SanityMujoco-v0', 'Environment name.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
config_flags.DEFINE_config_file(
    'config',
    'configs/droq_config.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)

# ==================== Eval Flags ====================
flags.DEFINE_integer('eval_episodes', 1,
                     'Number of episodes used for evaluation.')

# ==================== Log / Save Flags ====================
flags.DEFINE_integer('log_interval', 1, 'Logging interval.')
flags.DEFINE_string("save_dir", "./saved", "Directory to save the model checkpoint and replay buffer.")
flags.DEFINE_string('project_name', 'a1-route-laura', 'wandb project name.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')

flags.DEFINE_boolean('save_eval_videos', False, 'Save videos during evaluation.')
flags.DEFINE_integer("eval_video_length_limit", 0, "Limit the length of evaluation videos.")
flags.DEFINE_integer("eval_video_interval", 1, "Interval to save videos during evaluation.")
flags.DEFINE_boolean('save_eval_rollouts', False, 'Save rollouts during evaluation.')

flags.DEFINE_boolean('launch_viewer', False, "Launch a windowed viewer for the off-screen rendered environment frames.")
flags.DEFINE_boolean('launch_target_viewer', False, "Launch a windowed viewer for joystick heading and target.")
# ========================================================

# ==================== Joystick Task Flags ====================
config_flags.DEFINE_config_file(
    'task_config',
    'task_configs/default.py',
    'File path to the task/control config parameters.',
    lock_config=False)
config_flags.DEFINE_config_file(
    'reset_agent_config',
    'configs/reset_config.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)
# ========================================================
def import_register():
    if "Real" in FLAGS.env_name:
        import rail_walker_gym.envs.register_real
    if "Mujoco" in FLAGS.env_name:
        import rail_walker_gym.envs.register_mujoco

def main(_):
    import_register()
    exp_name = FLAGS.env_name
    exp_name += f'_s{FLAGS.seed}'
    #exp_name += f'_maxgr{FLAGS.config.max_gradient_norm:.2f}' if FLAGS.config.max_gradient_norm is not None else ''

    # ==================== Setup WandB ====================
    wandb.init(project=FLAGS.project_name, dir=os.getenv('WANDB_LOGDIR'))
    wandb.run.name = "Eval-" + exp_name
    wandb.config.update(FLAGS)
    
    # ==================== Setup Environment ====================
    eval_env : gym.Env = gym.make(FLAGS.env_name)
    
    try:
        eval_joystick_policy : rail_walker_interface.JoystickPolicy = eval_env.joystick_policy
    except:
        eval_joystick_policy = None
        print("Failed to get joystick policy from environment.")

    eval_env_need_render = FLAGS.launch_viewer or FLAGS.save_eval_videos

    if eval_joystick_policy is not None and eval_joystick_policy.robot.is_real_robot and eval_env_need_render:
        from rail_walker_gym.envs.wrappers.real_render import RealRenderWrapperWithSim
        eval_env = RealRenderWrapperWithSim(eval_env)

    if FLAGS.launch_viewer:
        eval_env = rail_walker_gym.envs.wrappers.RenderViewerWrapper(eval_env)
    
    if FLAGS.launch_target_viewer:
        env = rail_walker_gym.envs.wrappers.JoystickTargetViewer(env)

    task_suffix, eval_env = apply_task_configs(eval_env, FLAGS.env_name, 0, FLAGS.task_config, FLAGS.reset_agent_config, True)
    exp_name += task_suffix
    wandb.run.name = "Eval-" + exp_name

    eval_env.seed(FLAGS.seed)

    if FLAGS.save_eval_rollouts:
        eval_env = rail_walker_gym.envs.wrappers.RolloutCollector(eval_env) # wrap environment to automatically collect rollout

    if FLAGS.save_eval_videos:
        eval_env = rail_walker_gym.envs.wrappers.WanDBVideoWrapper(eval_env, record_every_n_steps=FLAGS.eval_video_interval, log_name="evaluation/video", video_length_limit=FLAGS.eval_video_length_limit) # wrap environment to automatically save video to wandb
        eval_env.enableWandbVideo = True
    
    observation, info = eval_env.reset(return_info=True)
    done = False
    # ==================== Setup Checkpointing ====================
    project_dir = os.path.join(FLAGS.save_dir, exp_name)
    initialize_project_log(project_dir)

    # ==================== Setup Learning Agent and Replay Buffer ====================
    agent_kwargs = dict(FLAGS.config)
    model_cls = agent_kwargs.pop('model_cls')

    agent = globals()[model_cls].create(
        FLAGS.seed, 
        eval_env.observation_space,
        eval_env.action_space,
        **agent_kwargs
    )

    agent_loaded_checkpoint_step, agent = load_latest_checkpoint(project_dir, agent, 0)
    if agent_loaded_checkpoint_step > 0:
        print(f"===================== Loaded checkpoint at step {agent_loaded_checkpoint_step} =====================")
    else:
        print(f"===================== No checkpoint found! =====================")
        print("Check directory:", project_dir)
        exit(0)
    # ==================== Start Eval ====================
    accumulated_info_dict = {}
    episode_counter = 0

    try:
        for i in tqdm.trange(0, 500000, initial=0, disable=not FLAGS.tqdm, smoothing=0.1):
            
            # Force steps in environment WanDBVideoWrapper to be the same as our training steps
            # Notice: we need to make sure that the very outside wrapper of env is WandbVideoWrapper, otherwise setting this will not work
            if FLAGS.save_eval_videos:
                eval_env.set_wandb_step(i)

            action = agent.eval_actions(observation)
            
            # Step the environment
            next_observation, reward, done, info = eval_env.step(action)
            truncated = "TimeLimit.truncated" in info and info['TimeLimit.truncated']
            if (not done) or truncated:
                mask = 1.0
            else:
                mask = 0.0

            if eval_joystick_policy is not None and eval_joystick_policy.robot.is_real_robot and (
                isinstance(eval_joystick_policy.robot.unwrapped(), rail_walker_interface.BaseWalkerWithJoystick) and
                eval_joystick_policy.robot.get_joystick_values()[-1][1] > 0.5
            ):
                print("right joystick up detected, pausing evaluation.")
                while (
                    isinstance(eval_joystick_policy.robot.unwrapped(), rail_walker_interface.BaseWalkerWithJoystick) and
                    eval_joystick_policy.robot.get_joystick_values()[-1][1] > 0.5
                ):
                    eval_joystick_policy.robot.receive_observation()
                    time.sleep(eval_joystick_policy.robot.control_timestep)
                print("right joystick up no longer detected, resuming training.")
                next_observation = eval_env.reset(return_info=False)
                done = False

            if 'TimeLimit.joystick_target_change' in info and info['TimeLimit.joystick_target_change']:
                next_observation, info = eval_env.reset(return_info=True)
                done=False

            observation = next_observation

            # Accumulate info to log
            for key in info.keys():
                if key in ['TimeLimit.truncated', 'TimeLimit.joystick_target_change', 'episode']:
                    continue
                value = info[key]
                if key not in accumulated_info_dict:
                    accumulated_info_dict[key] = [value]
                else:
                    accumulated_info_dict[key].append(value)

            if i % FLAGS.log_interval == 0:
                for k in accumulated_info_dict.keys():
                    v = accumulated_info_dict[k]
                    if v is None or len(v) <= 0:
                        continue
                    if k in ['fall_count','traversible_finished_lap_count']:
                        to_log = v[-1]
                    else:
                        to_log = np.mean(v)
                    wandb.log({'evaluation/' + str(k): to_log}, step=i)
                accumulated_info_dict = {}

            if done:
                for k, v in info['episode'].items():
                    decode = {'r': 'return', 'l': 'length', 't': 'time'}
                    wandb.log({f'evaluation/episode_{decode[k]}': v}, step=i)

                observation, info = eval_env.reset(return_info=True)
                done = False
                if FLAGS.save_eval_rollouts:
                    # Since we clear the collected rollouts after saving, we don't need to worry about saving episode rollout
                    save_rollout(project_dir, i, False, eval_env.collected_rollouts)
                    eval_env.collected_rollouts.clear()
                episode_counter += 1
            
            if episode_counter >= FLAGS.eval_episodes:
                break
    except KeyboardInterrupt:
        print("KeyboardInterrupt, please wait for clean up...")
        import traceback
        traceback.print_exc()
    finally:
        print("======================== Cleaning up ========================")
        # ==================== Save Final Checkpoint ====================
        if FLAGS.save_eval_rollouts and len(eval_env.collected_rollouts) > 0:
            save_rollout(project_dir, i, False, eval_env.collected_rollouts)
        if FLAGS.save_eval_videos:
            eval_env._terminate_record()
        eval_env.close()
        print("======================== Done ========================")

if __name__ == '__main__':
    app.run(main)
