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
from ml_collections import config_flags
import pickle
# from dm_control import viewer
import typing
import time
import mujoco
from natsort import natsorted 
import shutil
import rail_walker_gym
import rail_walker_interface
import matplotlib.pyplot as plt

from checkpoint_util import initialize_project_log, load_latest_checkpoint, save_checkpoint, load_latest_replay_buffer, load_latest_additional_replay_buffer, load_replay_buffer_file, save_replay_buffer, save_rollout
from eval_util import evaluate, evaluate_route_following, log_visitation, update_with_delay_with_mixed_buffers
from task_config_util import apply_task_configs
from action_curriculum_util import get_action_curriculum_planner_linear, get_action_curriculum_planner_quadratic


FLAGS = flags.FLAGS

# ==================== Training Flags ====================
flags.DEFINE_string('env_name', 'Go1SanityMujoco-v0', 'Environment name.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(5e5), 'Number of training steps.')
flags.DEFINE_integer('start_training', int(1000),
                     'Number of training steps to start training.')
flags.DEFINE_integer('utd_ratio', 20, 'Update to data ratio.')
flags.DEFINE_integer('action_curriculum_steps', -1, 'Number of steps to go to full action range.')
flags.DEFINE_float('action_curriculum_start', 0.1, 'Smallest action range')
flags.DEFINE_float('action_curriculum_end', 1.0, 'Largest action range')
flags.DEFINE_boolean('action_curriculum_linear', True, 'Use linear action curriculum, otherwise quadratic.')
flags.DEFINE_float('action_curriculum_exploration_eps', 0.0, 'Full Action Range Exploration epsilon for action curriculum.')
flags.DEFINE_boolean('load_buffer', False, 'Load replay buffer.')
flags.DEFINE_boolean('variable_friction', False, 'Use different friction for each quadrant of the heightfield.')
config_flags.DEFINE_config_file(
    'config',
    'configs/droq_config.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)
flags.DEFINE_integer('actor_delay', 20, 'Number of critic updates per actor update.')
flags.DEFINE_string("additional_buffer", None, "Additional replay buffer to load.")
flags.DEFINE_float("additional_buffer_prior", 0.5, "Prior probability of sampling from additional buffer.")
flags.DEFINE_float('terrain_scale', 0.5, 'Scale of generated terrain.')
flags.DEFINE_integer('reset_interval', -1, 'Number of steps between resets. -1 for no resets.')
flags.DEFINE_boolean('reset_agent', False, 'Whether to reset the learning agent.')
flags.DEFINE_boolean('reset_curriculum', True, 'Whether to reset the curriculum.')
flags.DEFINE_string('reset_criterion', 'time', 'Criterion for resetting the curriculum.')
flags.DEFINE_float('threshold', 1.0, 'Threshold to choose when to reset the curriculum (dynamics based only)')

# ==================== Eval Flags ====================
flags.DEFINE_string("eval_env_name", "", "Environment name for evaluation. If empty, use training environment name.")
flags.DEFINE_integer('eval_episodes', 1,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('eval_interval', 10000, 'Eval interval.')

# ==================== Log / Save Flags ====================
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer("save_interval", 10000, "Save interval.")
flags.DEFINE_string("save_dir", "./saved", "Directory to save the model checkpoint and replay buffer.")
flags.DEFINE_boolean('save_buffer', True, 'Save replay buffer for future training.')
flags.DEFINE_boolean('save_old_buffers', False, 'Keep replay buffers in previous steps.')
flags.DEFINE_string('project_name', 'a1-route-laura', 'wandb project name.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')

flags.DEFINE_boolean('save_eval_videos', False, 'Save videos during evaluation.')
flags.DEFINE_integer("eval_video_length_limit", 0, "Limit the length of evaluation videos.")
flags.DEFINE_boolean('save_eval_rollouts', False, 'Save rollouts during evaluation.')
flags.DEFINE_boolean('save_training_videos', False, 'Save videos during training.')
flags.DEFINE_integer("training_video_length_limit", 0, "Limit the length of training videos.")
flags.DEFINE_integer("training_video_interval", 3000, "Interval to save training videos.")
flags.DEFINE_boolean('save_training_rollouts', False, 'Save rollouts during training.')

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
flags.DEFINE_integer("leg_dropout_step", -1, "Step to start leg dropout.")
# ========================================================

def import_register():
    if "Real" in FLAGS.env_name:
        import rail_walker_gym.envs.register_real
    if "Mujoco" in FLAGS.env_name:
        import rail_walker_gym.envs.register_mujoco

def compute_priority_weights(x, const, scale):
    indices = np.arange(x)
    t = indices - x
    weights = np.exp(t / scale) + const
    weights = weights / np.sum(weights)
    return weights

def initialize_agent(seed, model_cls, obs_space, act_space, **agent_kwargs):
    agent = globals()[model_cls].create(
        seed, 
        obs_space,
        act_space, 
        **agent_kwargs
    )
    return agent

def get_dropout_apply_action(robot : rail_walker_interface.BaseWalker, dropout_leg_idx : typing.List[int]):
    original_fn = robot.apply_action
    def dropout_apply_action(action : np.ndarray) -> bool:
        action = action.copy()
        action[dropout_leg_idx] = robot.joint_qpos_init[dropout_leg_idx]
        return original_fn(action)
    return dropout_apply_action

def main(_):
    import_register()
    exp_name = FLAGS.env_name
    exp_name += f'_s{FLAGS.seed}_maxgr{FLAGS.config.max_gradient_norm:.2f}'
    
    if FLAGS.reset_interval > 0:
        exp_name += f'_reset{FLAGS.reset_interval}'

    # else:
    #     exp_name += '_stepwise'

    if FLAGS.config.exterior_linear_c > 0.0:
        exp_name += f'_extlin{FLAGS.config.exterior_linear_c:.2f}'
    elif FLAGS.config.exterior_quadratic_c > 0.0:
        exp_name += f'_extquad{FLAGS.config.exterior_quadratic_c:.2f}'
        
    if FLAGS.config.interior_quadratic_c > 0.0:
        exp_name += f'_intquad{FLAGS.config.interior_quadratic_c:.2f}'
    
    if FLAGS.variable_friction:
        exp_name += '_varfriction'
        
    if FLAGS.action_curriculum_steps > 0:
        exp_name += f'_ac{FLAGS.action_curriculum_steps}_{FLAGS.action_curriculum_start}to{FLAGS.action_curriculum_end}'
        if FLAGS.action_curriculum_linear:
            exp_name += '_linearly'
        else:
            exp_name += '_quad'

    if "Bumpy" in FLAGS.env_name:
        exp_name += f"_ts{FLAGS.terrain_scale}"
    # ==================== Setup WandB ====================
    wandb.init(project=FLAGS.project_name, dir=os.getenv('WANDB_LOGDIR'))
    wandb.run.name = exp_name
    wandb.config.update(FLAGS)
    
    # ==================== Setup Environment ====================
    env : gym.Env = gym.make(FLAGS.env_name)
    
    if hasattr(env.unwrapped, "task") and hasattr(env.unwrapped.task, "_floor"):
        env.unwrapped.task._floor.terrain_smoothness = 0.5
        env.unwrapped.task._floor.terrain_scale = FLAGS.terrain_scale

    try:
        joystick_policy : rail_walker_interface.JoystickPolicy = env.joystick_policy
    except:
        joystick_policy = None
        print("Failed to get joystick policy from environment.")
    
    env_need_render = FLAGS.launch_viewer or FLAGS.save_training_videos

    if joystick_policy is not None and joystick_policy.robot.is_real_robot and env_need_render:
        from rail_walker_gym.envs.wrappers.real_render import RealRenderWrapperWithSim
        env = RealRenderWrapperWithSim(env, need_render=env_need_render)

    if FLAGS.launch_viewer:
        env = rail_walker_gym.envs.wrappers.RenderViewerWrapper(env)
    
    if FLAGS.launch_target_viewer:
        env = rail_walker_gym.envs.wrappers.JoystickTargetViewer(env)

    if FLAGS.variable_friction:
        friction_list = [0.8, 0.1, 1.2, 0.2, 0.4, 1.0, 0.8, 0.1, 1.2, 0.2, 0.4, 1.0]
        segments = []
        for f in friction_list:
            segments.append(np.ones(FLAGS.action_curriculum_steps)*f)
        friction_schedule = np.concatenate(segments)

    task_suffix, env = apply_task_configs(env, FLAGS.env_name, FLAGS.max_steps, FLAGS.task_config, FLAGS.reset_agent_config, False)
    # exp_name += task_suffix
    wandb.run.name = exp_name

    if FLAGS.eval_interval <= 0:
        eval_env = env
        eval_joystick_policy = joystick_policy
    else:
        eval_env_name = FLAGS.eval_env_name if FLAGS.eval_env_name else FLAGS.env_name
        eval_env : gym.Env = gym.make(eval_env_name)
        try:
            eval_joystick_policy : rail_walker_interface.JoystickPolicy = eval_env.joystick_policy
        except:
            eval_joystick_policy = None
            print("Failed to get joystick policy from eval environment.")
    
        eval_env_need_render = FLAGS.save_eval_videos
        if eval_joystick_policy is not None and eval_joystick_policy.robot.is_real_robot and eval_env_need_render:
            from rail_walker_gym.envs.wrappers.real_render import RealRenderWrapperWithSim
            eval_env = RealRenderWrapperWithSim(eval_env)
        
        _, eval_env = apply_task_configs(eval_env, FLAGS.env_name, FLAGS.max_steps, FLAGS.task_config, FLAGS.reset_agent_config, True)
    
    env.seed(FLAGS.seed)
    eval_env.seed(FLAGS.seed)

    if FLAGS.save_training_rollouts:
        env = rail_walker_gym.envs.wrappers.RolloutCollector(env) # wrap environment to automatically collect rollout

    if FLAGS.save_training_videos:
        env = rail_walker_gym.envs.wrappers.WanDBVideoWrapper(env, record_every_n_steps=FLAGS.training_video_interval, video_length_limit=FLAGS.training_video_length_limit) # wrap environment to automatically save video to wandb
        env.enableWandbVideo = True
    
    observation, info = env.reset(return_info=True)
    # import ipdb; ipdb.set_trace()
    done = False
    # ==================== Setup Checkpointing ====================
    project_dir = os.path.join(FLAGS.save_dir, exp_name)
    initialize_project_log(project_dir)

    # ==================== Setup Learning Agent and Replay Buffer ====================
    agent_kwargs = dict(FLAGS.config)
    model_cls = agent_kwargs.pop('model_cls')

    agent = initialize_agent(FLAGS.seed, model_cls, env.observation_space, env.action_space, **agent_kwargs)
    agent_loaded_checkpoint_step, agent = load_latest_checkpoint(project_dir, agent, 0)

    if agent_loaded_checkpoint_step > 0:
        print(f"===================== Loaded checkpoint at step {agent_loaded_checkpoint_step} =====================")
    else:
        print(f"===================== No checkpoint found. =====================")

    replay_buffer = None
    if FLAGS.load_buffer:
        rb_load = load_latest_replay_buffer(project_dir)
        if rb_load is not None:
            replay_buffer_loaded_step, replay_buffer = rb_load
            print(f"===================== Loaded replay buffer at step {replay_buffer_loaded_step} =====================")

    additional_replay_buffer = None
    if FLAGS.additional_buffer is not None:
        try:
            additional_replay_buffer = load_replay_buffer_file(FLAGS.additional_buffer)
            print(f"===================== Loaded additional replay buffer from {FLAGS.additional_buffer} =====================")
        except:
            additional_replay_buffer = load_latest_additional_replay_buffer(project_dir)
            if additional_replay_buffer is not None:
                print(f"===================== Loaded additional replay buffer from additional_buffers/ =====================")
        
    if replay_buffer is None:
        replay_buffer = ReplayBuffer(env.observation_space, env.action_space,
                                 FLAGS.max_steps)
        replay_buffer_loaded_step = 0
    
    replay_buffer.seed(FLAGS.seed)


    # ==================== Start Training ====================
    if FLAGS.leg_dropout_step > 0 and agent_loaded_checkpoint_step > FLAGS.leg_dropout_step:
        print("Leg dropout step reached, dropping out leg 2.")
        joystick_policy.robot.apply_action = get_dropout_apply_action(
            joystick_policy.robot,
            [2]
        )
    
    accumulated_info_dict = {}
    jax_jitted_actions = False
    jax_jitted_rb = False

    tracked_locations = []

    if FLAGS.action_curriculum_steps <= 0:
        action_curriculum_planner = lambda x: (-FLAGS.action_curriculum_end, FLAGS.action_curriculum_end)
    elif FLAGS.action_curriculum_linear:
        action_curriculum_planner = get_action_curriculum_planner_linear(FLAGS.action_curriculum_steps, FLAGS.action_curriculum_start, FLAGS.action_curriculum_end)
    else:
        action_curriculum_planner = get_action_curriculum_planner_quadratic(FLAGS.action_curriculum_steps, FLAGS.action_curriculum_start, FLAGS.action_curriculum_end)

    actions = []
    values = []
    td_errors = []
    dyn_errors = []
    # for automated curriculum
    ema = None
    dynamics_loss_mean, M2, estimate_count = 0.0, 0.0, 0

    def update_dynerror_statistics(value):
        nonlocal estimate_count, dynamics_loss_mean, M2
        estimate_count += 1
        delta = value - dynamics_loss_mean
        dynamics_loss_mean += delta / estimate_count
        delta2 = value - dynamics_loss_mean
        M2 += delta * delta2

    falls = []
    
    start_time = time.time()
    try:
        for i in tqdm.trange(agent_loaded_checkpoint_step, FLAGS.max_steps + agent_loaded_checkpoint_step, initial=agent_loaded_checkpoint_step, disable=not FLAGS.tqdm, smoothing=0.1):
            if FLAGS.variable_friction:
                env.unwrapped.task.foot_friction = friction_schedule[i]
            # Force steps in environment WanDBVideoWrapper to be the same as our training steps
            # Notice: we need to make sure that the very outside wrapper of env is WandbVideoWrapper, otherwise setting this will not work
            if FLAGS.save_training_videos:
                env.set_wandb_step(i)
            
            # Hack for Jax Jit
            if i >= FLAGS.start_training and not jax_jitted_actions:
                jax_jitted_actions = True
                print("==================== Jitting Jax Actions ====================")
                if joystick_policy is not None:
                    for _ in range(10):
                        joystick_policy.robot.apply_action(joystick_policy.robot.joint_qpos_init)
                
                for _ in range(5):
                    action, agent = agent.sample_actions(env.observation_space.sample())

                print("==================== Done Jitting Jax ====================")
                hack_jit_obs, hack_jit_info = env.reset(return_info=True)
                observation = hack_jit_obs
                if FLAGS.save_training_videos:
                    env.set_wandb_step(i)
                
            if i >= FLAGS.start_training + agent_loaded_checkpoint_step - replay_buffer_loaded_step and not jax_jitted_rb:
                jax_jitted_rb = True
                print("==================== Jitting Jax updates ====================")
                if joystick_policy is not None:
                    for _ in range(20):
                        joystick_policy.robot.apply_action(joystick_policy.robot.joint_qpos_init)
                
                if hasattr(joystick_policy.robot,"hibernate") and hasattr(joystick_policy.robot,"cancel_hibernate"):
                    joystick_policy.robot.hibernate()
                    
                for _ in tqdm.trange(3, disable=not FLAGS.tqdm):
                    if additional_replay_buffer is None:
                        batch = replay_buffer.sample(FLAGS.batch_size * FLAGS.utd_ratio)
                        agent, update_info = agent.update_with_delay(batch, FLAGS.utd_ratio, output_range=action_curriculum_planner(i), actor_delay=FLAGS.actor_delay)
                    else:
                        agent, update_info = update_with_delay_with_mixed_buffers(
                            additional_replay_buffer,
                            replay_buffer,
                            FLAGS.batch_size,
                            FLAGS.utd_ratio,
                            agent,
                            FLAGS.additional_buffer_prior,
                            FLAGS.actor_delay,
                            action_curriculum_planner(i)
                        )

                if hasattr(joystick_policy.robot,"hibernate") and hasattr(joystick_policy.robot,"cancel_hibernate"):
                    joystick_policy.robot.cancel_hibernate()

                print("==================== Done Jitting Jax ====================")
                hack_jit_obs, hack_jit_info = env.reset(return_info=True)
                observation = hack_jit_obs
                if FLAGS.save_training_videos:
                    env.set_wandb_step(i)
            
            if i < FLAGS.start_training:
                action = env.action_space.sample()
                planned_action_range = action_curriculum_planner(i)
                # planned_action_range = curriculum_planners[which_agent](i)
                if planned_action_range is not None:
                    multiplier = planned_action_range[1]
                    if not(FLAGS.action_curriculum_exploration_eps <= 0.0 or np.random.rand() >= FLAGS.action_curriculum_exploration_eps):
                        multiplier = min(1.0, multiplier * 1.5)
                    
                    action = action * multiplier
                        
            else:
                action, agent = agent.sample_actions(observation)
                
            actions.append(action)
            
            # Step the environment
            next_observation, reward, done, info = env.step(action)
            truncated = "TimeLimit.truncated" in info and info['TimeLimit.truncated']
            if (not done) or truncated:
                mask = 1.0
            else:
                mask = 0.0

            if joystick_policy is not None and joystick_policy.robot.is_real_robot and ((
                #instance(joystick_policy.robot.unwrapped(), rail_walker_interface.BaseWalkerWithFootContact) and 
                #np.count_nonzero(joystick_policy.robot.get_foot_contact()) <= 1
            ) or (
                isinstance(joystick_policy.robot.unwrapped(), rail_walker_interface.BaseWalkerWithJoystick) and
                joystick_policy.robot.get_joystick_values()[-1][1] > 0.5
            )):
                print("Lifted foot or right joystick up detected, not inserting into replay buffer.")
                while (
                    isinstance(joystick_policy.robot.unwrapped(), rail_walker_interface.BaseWalkerWithFootContact) and 
                    np.count_nonzero(joystick_policy.robot.get_foot_contact()) <= 1
                ) or (
                    isinstance(joystick_policy.robot.unwrapped(), rail_walker_interface.BaseWalkerWithJoystick) and
                    joystick_policy.robot.get_joystick_values()[-1][1] > 0.5
                ):
                    joystick_policy.robot.receive_observation()
                    time.sleep(joystick_policy.robot.control_timestep)
                print("Foot down or right joystick down detected, resuming training.")
                next_observation = env.reset(return_info=False)
                done = False
            elif (
                joystick_policy is not None and
                isinstance(joystick_policy.robot, rail_walker_interface.BaseWalkerWithJointTemperatureSensor) and
                np.any(joystick_policy.robot.get_joint_temperature_celsius() > 60)
            ):
                print("Overheated joint detected, resetting robot")
                if hasattr(joystick_policy.robot,"hibernate") and hasattr(joystick_policy.robot,"cancel_hibernate"):
                    joystick_policy.robot.hibernate()
                    input("Press enter to continue")
                    joystick_policy.robot.cancel_hibernate()
                else:
                    input("Press enter to continue")
                next_observation, info = env.reset(return_info=True)
                done = False
            else:
                transition_dict = dict(
                    observations=observation,
                    actions=action,
                    rewards=reward,
                    masks=mask,
                    dones=done,
                    next_observations=next_observation
                )
                # falls.append(info['fall_count'])
                # Insert the transition in the replay buffer
                replay_buffer.insert(transition_dict)
            
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

            # Update the agent
            if i - agent_loaded_checkpoint_step >= FLAGS.start_training - replay_buffer_loaded_step:
                # Relabel batch
                if FLAGS.task_config.relabel_update_fraction > 0.0 and hasattr(env, "get_relabel_replay_buffer"):
                    batch_sample_num = int(round(FLAGS.batch_size * (1.0 - FLAGS.task_config.relabel_update_fraction)))
                    batch = replay_buffer.sample(batch_sample_num)
                    relabel_replay_buffer : ReplayBuffer = env.get_relabel_replay_buffer()
                    batch_relabel = relabel_replay_buffer.sample(FLAGS.batch_size * FLAGS.utd_ratio - batch_sample_num)
                    batch = dict(
                        observations=np.concatenate([batch['observations'], batch_relabel['observations']], axis=0),
                        actions=np.concatenate([batch['actions'], batch_relabel['actions']], axis=0),
                        rewards=np.concatenate([batch['rewards'], batch_relabel['rewards']], axis=0),
                        masks=np.concatenate([batch['masks'], batch_relabel['masks']], axis=0),
                        dones=np.concatenate([batch['dones'], batch_relabel['dones']], axis=0),
                        next_observations=np.concatenate([batch['next_observations'], batch_relabel['next_observations']], axis=0),
                    )
                else:
                    batch = replay_buffer.sample(FLAGS.batch_size * FLAGS.utd_ratio)
                
                
                agent, update_info = agent.update_with_delay(batch, FLAGS.utd_ratio, output_range=action_curriculum_planner(i), actor_delay=FLAGS.actor_delay)
                # update_dynerror_statistics(update_info['dynamics_loss'])
                
                if i % FLAGS.log_interval == 0:
                    wandb.log({f'training/fps': i/(time.time() - start_time)}, step=i)
                    # for all keys in each update_info, add an agent index to the key
                    for k, v in update_info.items():
                        wandb.log({f'training/agent_0/{k}': v.item()}, step=i)
                    # wandb.log({'training/dynamics_surprise': dyn_loss.item()}, step=i)

            
            if i % FLAGS.save_interval == 0 and i > agent_loaded_checkpoint_step:
                save_checkpoint(os.path.join(project_dir), i, agent)

                if i % (FLAGS.log_interval * 10) == 0 and i > agent_loaded_checkpoint_step and FLAGS.save_buffer:
                    save_replay_buffer(project_dir, i, replay_buffer, not FLAGS.save_old_buffers)

            if i % FLAGS.log_interval == 0 and i > agent_loaded_checkpoint_step:
                for k in accumulated_info_dict.keys():
                    v = accumulated_info_dict[k]
                    if v is None or len(v) <= 0:
                        continue
                    
                    if k == '2d_location':
                        # log_visitation(env, v, i, project_dir)
                        tracked_locations.extend(v)
                        tracked_locations = tracked_locations[-FLAGS.save_interval:]
                    else:
                        if k in ['fall_count', 'traversible_finished_lap_count']:
                            to_log = v[-1]
                        else:
                            to_log = np.mean(v)
                        wandb.log({'training/' + str(k): to_log}, step=i)
                accumulated_info_dict = {}

                ################## Skip this for now ###################

                # Compute histogram
                actions = np.array(actions).flatten()
                plt.clf()
                
                hist, bins = np.histogram(actions, bins='auto')
                fig, ax = plt.subplots()
                ax.bar(bins[:-1], hist, width=np.diff(bins), align='edge')
                ax.set_xlabel('Action')
                ax.set_ylabel('Frequency')
                ax.set_title('Histogram of Actions')
                fig.canvas.draw()
                img = np.array(fig.canvas.renderer.buffer_rgba())[:,:,0:3]
                wandb.log({"training/action_hist": wandb.Image(img)}, step=i)
                
                actions = []

            if FLAGS.leg_dropout_step > 0 and i == FLAGS.leg_dropout_step:
                print("Leg dropout step reached, dropping out leg 2.")
                joystick_policy.robot.apply_action = get_dropout_apply_action(
                    joystick_policy.robot,
                    [2]
                )

            if done:
                if 'episode' not in info:
                    import ipdb; ipdb.set_trace()
                for k, v in info['episode'].items():
                    decode = {'r': 'return', 'l': 'length', 't': 'time'}
                    wandb.log({f'training/episode_{decode[k]}': v}, step=i)
                
                if FLAGS.reset_curriculum and FLAGS.reset_criterion == "dynamics_error":
                    recent_batch_indx = np.arange(len(replay_buffer) - FLAGS.batch_size, len(replay_buffer))
                    recent_batch = replay_buffer.sample(FLAGS.batch_size, indx=recent_batch_indx)
                    
                    dyn_errors.append(agent.compute_dynamics_surprise(recent_batch))            

                    wandb.log({f'training/dyn_surprise': np.mean(dyn_errors)}, step=i)
                    
                    if ema is None:
                        ema = np.mean(dyn_errors)
                    else:
                        ema = 0.5 * np.mean(dyn_errors) + .5 * ema
                    dyn_errors = []

                    if ema > FLAGS.threshold:
                        curr_planner = action_curriculum_planner(i)[1]
                        shrink_by_half = curr_planner * 0.5
                        if FLAGS.action_curriculum_steps <= 0:
                            action_curriculum_planner = lambda x: (-FLAGS.action_curriculum_end, FLAGS.action_curriculum_end)
                        elif FLAGS.action_curriculum_linear:
                            start_value = max(FLAGS.action_curriculum_start, shrink_by_half)
                            action_curriculum_planner = get_action_curriculum_planner_linear(FLAGS.action_curriculum_steps, start_value, FLAGS.action_curriculum_end, start_step=i)
                        else:
                            action_curriculum_planner = get_action_curriculum_planner_quadratic(FLAGS.action_curriculum_steps, FLAGS.action_curriculum_start, FLAGS.action_curriculum_end)
                        # # reset the estimates
                        # dynamics_loss_mean, M2, estimate_count = 0.0, 0.0, 0
                        ema = None

                        if FLAGS.reset_agent:
                            agent = initialize_agent(FLAGS.seed, model_cls, env.observation_space, env.action_space, agent_kwargs)
                        
                planned_action_range = action_curriculum_planner(i)
                # planned_action_range = curriculum_planners[agent_i](i)
                if planned_action_range is not None:
                    low = np.around(planned_action_range[0], 2)
                    high = np.around(planned_action_range[1], 2)
                    wandb.log({'training/action_range_low': low}, step=i)
                    wandb.log({'training/action_range_high': high}, step=i)

                observation, info = env.reset(return_info=True)
                done = False
                if FLAGS.save_training_rollouts:
                    # Since we clear the collected rollouts after saving, we don't need to worry about saving episode rollout
                    save_rollout(project_dir, i, True, env.collected_rollouts)
                    env.collected_rollouts.clear()
        
            if FLAGS.reset_interval > 0 and i % FLAGS.reset_interval == 0 and FLAGS.reset_curriculum and i > 0 and FLAGS.reset_criterion == 'time':
                if FLAGS.action_curriculum_steps <= 0:
                    action_curriculum_planner = lambda x: (-FLAGS.action_curriculum_end, FLAGS.action_curriculum_end)
                elif FLAGS.action_curriculum_linear:
                    action_curriculum_planner = get_action_curriculum_planner_linear(FLAGS.action_curriculum_steps, FLAGS.action_curriculum_start, FLAGS.action_curriculum_end, start_step=i)
                else:
                    action_curriculum_planner = get_action_curriculum_planner_quadratic(FLAGS.action_curriculum_steps, FLAGS.action_curriculum_start, FLAGS.action_curriculum_end)
                
                if FLAGS.reset_agent:
                    agent = initialize_agent(FLAGS.seed, model_cls, env.observation_space, env.action_space, agent_kwargs)

    except KeyboardInterrupt:
        print("KeyboardInterrupt, please wait for clean up...")
        import traceback
        traceback.print_exc()
    except:
        import traceback
        traceback.print_exc()
    finally:
        print("======================== Cleaning up ========================")
        # ==================== Save Final Checkpoint ====================
        if i > agent_loaded_checkpoint_step + FLAGS.start_training:
            save_checkpoint(project_dir, i, agent)
            if FLAGS.save_buffer:
                save_replay_buffer(project_dir, i, replay_buffer, not FLAGS.save_old_buffers)
        if FLAGS.save_training_rollouts and len(env.collected_rollouts) > 0:
            save_rollout(project_dir, i, True, env.collected_rollouts)
        if FLAGS.save_training_videos:
            env._terminate_record()
        
        env.close()
        if eval_env is not env:
            eval_env.close()
        print("======================== Done ========================")

if __name__ == '__main__':
    app.run(main)