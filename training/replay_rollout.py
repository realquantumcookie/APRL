import gym
import numpy as np
import tqdm
from absl import app, flags
from jaxrl5.wrappers import wrap_gym
import rail_walker_gym
import rail_walker_interface
import wandb
import os
from checkpoint_util import load_rollout_at_step, save_rollout
from ml_collections import config_flags
from task_config_util import apply_task_configs
import time

# ==================== Replay Flags ====================
flags.DEFINE_string("replay_npz", None, "Path to the replay file to load.")
flags.DEFINE_string('env_name', 'Go1SanityMujoco-v0', 'Environment name.')
flags.DEFINE_integer('seed', 10, 'Random seed.')

# ==================== Log/Save Flags ====================
flags.DEFINE_string('project_name', 'a1-route-laura', 'wandb project name.')
flags.DEFINE_bool("save_video", False, "Save video of replay interactions.")
flags.DEFINE_bool("launch_viewer", False, "Launch a windowed viewer for the off-screen rendered environment frames.")
flags.DEFINE_boolean('launch_target_viewer', False, "Launch a windowed viewer for joystick heading and target.")
flags.DEFINE_string("store_npz", None, "Path to the replay file to store.")
flags.DEFINE_integer('log_interval', 1, 'Logging interval.')

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

FLAGS = flags.FLAGS

def import_register():
    if "Real" in FLAGS.env_name:
        import rail_walker_gym.envs.register_real
    if "Mujoco" in FLAGS.env_name:
        import rail_walker_gym.envs.register_mujoco

def main(_):
    import_register()
    exp_name = FLAGS.env_name
    exp_name += f'_s{FLAGS.seed}'

    # ==================== Setup WandB ====================
    wandb.init(project=FLAGS.project_name, dir=os.getenv('WANDB_LOGDIR'))
    wandb.run.name = "Replay-" + exp_name
    wandb.config.update(FLAGS)
    
    # ==================== Setup Environment ====================
    env : gym.Env = gym.make(FLAGS.env_name)

    try:
        joystick_policy : rail_walker_interface.JoystickPolicy = env.joystick_policy
    except:
        joystick_policy = None
        print("Failed to get joystick policy from environment.")

    env_need_render = FLAGS.launch_viewer or FLAGS.save_video

    if joystick_policy is not None and joystick_policy.robot.is_real_robot and env_need_render:
        from rail_walker_gym.envs.wrappers.real_render import RealRenderWrapperWithSim
        env = RealRenderWrapperWithSim(env)

    if FLAGS.launch_viewer:
        env = rail_walker_gym.envs.wrappers.RenderViewerWrapper(env)
    

    if FLAGS.launch_target_viewer:
        env = rail_walker_gym.envs.wrappers.JoystickTargetViewer(env)

    task_suffix, env = apply_task_configs(env, FLAGS.env_name, 0, FLAGS.task_config, FLAGS.reset_agent_config, True)
    exp_name += task_suffix
    wandb.run.name = "Replay" + exp_name
    
    env.seed(FLAGS.seed)

    if FLAGS.store_npz is not None:
        env = rail_walker_gym.envs.wrappers.RolloutCollector(env) # wrap environment to automatically collect rollout

    if FLAGS.save_video:
        env = rail_walker_gym.envs.wrappers.WanDBVideoWrapper(env, record_every_n_steps=-1) # wrap environment to automatically save video to wandb
        env.enableWandbVideo = True
    
    observation, info = env.reset(return_info=True)
    done = False

    # ==================== Setup Rollout ====================
    to_replay = rail_walker_gym.envs.wrappers.rollout_collect.Rollout.import_npz(
        FLAGS.replay_npz
    )

    input()
    # ==================== Start Training ====================
    accumulated_info_dict = {}
    try:
        for i in tqdm.trange(0, len(to_replay), initial=0, smoothing=0.1):
            # Force steps in environment WanDBVideoWrapper to be the same as our training steps
            # Notice: we need to make sure that the very outside wrapper of env is WandbVideoWrapper, otherwise setting this will not work
            if FLAGS.save_video:
                env.set_wandb_step(i)

            # Get the action from the replay
            action = to_replay.actions[i]
            
            # Step the environment
            next_observation, reward, done, info = env.step(action)
            truncated = "TimeLimit.truncated" in info and info['TimeLimit.truncated']
            if (not done) or truncated:
                mask = 1.0
            else:
                mask = 0.0

            if joystick_policy is not None and joystick_policy.robot.is_real_robot and (
                isinstance(joystick_policy.robot.unwrapped(), rail_walker_interface.BaseWalkerWithJoystick) and
                joystick_policy.robot.get_joystick_values()[-1][1] > 0.5
            ):
                print("right joystick up detected, pausing rollout replay.")
                while (
                    isinstance(joystick_policy.robot.unwrapped(), rail_walker_interface.BaseWalkerWithJoystick) and
                    joystick_policy.robot.get_joystick_values()[-1][1] > 0.5
                ):
                    joystick_policy.robot.receive_observation()
                    time.sleep(joystick_policy.robot.control_timestep)
                print("right joystick up no longer detected, resuming training.")
                next_observation = env.reset(return_info=False)
                done = False

            if 'TimeLimit.joystick_target_change' in info and info['TimeLimit.joystick_target_change']:
                next_observation, info = env.reset(return_info=True)
                done = False
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

                    if k in ['fall_count', 'traversible_finished_lap_count']:
                        to_log = v[-1]
                    else:
                        to_log = np.mean(v)
                    wandb.log({'replay/' + str(k): to_log}, step=i)
                accumulated_info_dict = {}

            if done:
                for k, v in info['episode'].items():
                    decode = {'r': 'return', 'l': 'length', 't': 'time'}
                    wandb.log({f'training/episode_{decode[k]}': v}, step=i)

                observation, info = env.reset(return_info=True)
                done = False
    except KeyboardInterrupt:
        print("KeyboardInterrupt, Please wait for clean up...")
        import traceback
        traceback.print_exc()
    finally:
        print("======================== Cleaning up ========================")
        # ==================== Save Collected Rollout ====================
        if FLAGS.store_npz is not None:
            env.collected_rollouts.export_npz(FLAGS.store_npz)
        if FLAGS.save_video:
            env._terminate_record()
        env.close()
        print("======================== Done ========================")

if __name__ == '__main__':
    app.run(main)