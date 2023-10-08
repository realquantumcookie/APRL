from rail_walker_interface import JoystickPolicy, JoystickEnvironment, BaseWalker, BaseWalkerInSim, ReadOnlyWalkerWrapper, JoystickEnvImpl
from typing import Tuple
import gym
import gym.wrappers
from rail_walker_gym.envs.wrappers.relabel_wrapper import RelabelAggregateWrapper, RelabelTargetProvider, SymmetricRelabelWrapper
from rail_walker_gym.joystick_policy.target_providers import JoystickPolicyAutoJoystickTargetProvider
from rail_walker_gym.envs.register_helper import common_post_process_wrapper
import rail_walker_gym.envs.wrappers as envwrappers
from rail_walker_gym.joystick_policy import JoystickPolicyStrictRewardProvider, JoystickPolicyForwardOnlyTargetProvider, JoystickPolicyResetterEarlyStopTruncationProvider, JoystickPolicyManualResetter
from rail_walker_gym.joystick_policy_mujoco import JoystickPolicyPointInSimResetter, JoystickPolicyLastPositionAndYawResetter
from rail_mujoco_walker import RailSimWalkerDMControl
from checkpoint_util import load_checkpoint_file
from jaxrl5.agents import SACLearner
from jaxrl5.agents.sac.sac_learner_wdynamics import SACLearnerWithDynamics
import copy

def _helper_apply_resetter_wrapper(
    env: JoystickEnvironment,
    task_configs,
    reset_agent_config
):
    robot = env.joystick_policy.robot
    joystick_policy = env.joystick_policy
    
    # change fixed point resetter to initializers
    to_move = []
    for resetter in joystick_policy.resetters:
        if isinstance(resetter, JoystickPolicyPointInSimResetter) or isinstance(resetter, JoystickPolicyLastPositionAndYawResetter):
            to_move.append(resetter)
    for resetter in to_move:
        joystick_policy.resetters.remove(resetter)
    joystick_policy.initializers.extend(to_move)
    if isinstance(env.joystick_policy.robot, BaseWalkerInSim):
        joystick_policy.resetters.append(JoystickPolicyLastPositionAndYawResetter())

    def _helper_wrap_reset_env_fn(env):
        env, _ = common_post_process_wrapper(
            env,
            task_configs.reset_policy_action_range,
            task_configs.reset_policy_filter_actions,
            task_configs.reset_policy_frame_stack,
            task_configs.reset_policy_action_history,
            task_configs.reset_policy_center_init_action,
        )
        return env

    reset_joystick_policy = JoystickPolicy(
        robot,
        JoystickPolicyStrictRewardProvider(),
        JoystickPolicyForwardOnlyTargetProvider(),
        [],
        [JoystickPolicyResetterEarlyStopTruncationProvider()],
        [],
        initializers=[],
        target_observable = None,
        enabled_observables=task_configs.reset_policy_joystick_enabled_observables
    )
    resetter_obs_space, resetter_act_space = envwrappers.ResetterPolicySupportedEnvironment.compute_obs_act_space_reset_agent(
        env, reset_joystick_policy, _helper_wrap_reset_env_fn, task_configs.reset_policy_action_range
    )
    reset_agent_config = dict(reset_agent_config)
    reset_agent_model_cls = reset_agent_config.pop('model_cls')
    reset_agent = globals()[reset_agent_model_cls].create(
        0, 
        resetter_obs_space,
        resetter_act_space, 
        **reset_agent_config
    )
    reset_agent = load_checkpoint_file(
        task_configs.reset_agent_checkpoint, 
        reset_agent
    )
    if isinstance(robot.unwrapped(), BaseWalkerInSim):
        fallback_resetter = JoystickPolicyLastPositionAndYawResetter()
        if isinstance(robot.unwrapped(),RailSimWalkerDMControl):
            intercept_reset = True
        else:
            intercept_reset = False
    else:
        fallback_resetter = JoystickPolicyManualResetter()
        intercept_reset = False

    
    env = envwrappers.ResetterPolicySupportedEnvironment(
        env, 
        reset_joystick_policy, 
        reset_agent, 
        max_seconds=task_configs.reset_policy_max_seconds,
        wrap_resetter_env_lambda=_helper_wrap_reset_env_fn,
        fallback_resetter=fallback_resetter,
        resetter_Kp=task_configs.reset_policy_Kp,
        resetter_Kd=task_configs.reset_policy_Kd,
        resetter_action_interpolation=task_configs.reset_policy_action_interpolation,
        resetter_action_scale=task_configs.reset_policy_action_range,
        resetter_power_protect_factor=task_configs.reset_policy_power_protect_factor,
        intercept_all_reset_calls=intercept_reset
    )
    return env
    
def _apply_task_configs(
    env: JoystickEnvironment,
    task_configs,
    reset_agent_config,
    is_relabel_task : bool = False
) -> Tuple[str, gym.Env]:
    task_suffix = ""
    joystick_policy : JoystickPolicy = env.joystick_policy
    joystick_policy.robot.Kp = task_configs.Kp
    joystick_policy.robot.Kd = task_configs.Kd
    
    task_suffix += "_lt" if task_configs.lock_target else "_nlt"
    joystick_policy.lock_target = task_configs.lock_target

    task_suffix += "_PD{:.2f},{:.2f}".format(task_configs.Kp, task_configs.Kd)

    joystick_policy.robot.action_interpolation = task_configs.action_interpolation
    task_suffix += "_ai" if task_configs.action_interpolation else "_nai"
    task_suffix += str(task_configs.filter_actions) if task_configs.filter_actions > 0 else ""

    joystick_policy.robot.limit_action_range = task_configs.action_range
    task_suffix += "_ar{:.2f}".format(task_configs.action_range)

    task_suffix += "_fs{}".format(task_configs.frame_stack)

    task_suffix += "_ciqa" if task_configs.center_init_action else "_nciqa"

    try:
        joystick_policy.robot.power_protect_factor = task_configs.power_protect_factor
        task_suffix += "_ppf{:.2f}".format(task_configs.power_protect_factor)
    except ImportError:
        pass
    
    if hasattr(joystick_policy.target_yaw_provider, "target_linear_velocity"):
        setattr(joystick_policy.target_yaw_provider, "target_linear_velocity", task_configs.rew_target_velocity)

    if hasattr(joystick_policy.reward_provider, "smooth_tdy_steps"):
        setattr(joystick_policy.reward_provider, "smooth_tdy_steps", task_configs.rew_smooth_change_in_tdy_steps)

    if not joystick_policy.target_yaw_provider.is_target_velocity_fixed():
        task_suffix += "_tvdyn"
        if hasattr(joystick_policy.target_yaw_provider,"min_target_velocity") and hasattr(joystick_policy.target_yaw_provider,"max_target_velocity"):
            setattr(joystick_policy.target_yaw_provider, "min_target_velocity", task_configs.min_target_velocity)
            setattr(joystick_policy.target_yaw_provider, "max_target_velocity", task_configs.max_target_velocity)
            task_suffix += "{:.2f},{:.2f}".format(task_configs.min_target_velocity, task_configs.max_target_velocity)
        if hasattr(joystick_policy.target_yaw_provider,"max_velocity_change_rate"):
            setattr(joystick_policy.target_yaw_provider, "max_velocity_change_rate", task_configs.max_velocity_change_rate)
            task_suffix += "-{:.2f}".format(task_configs.max_velocity_change_rate)
    else:
        task_suffix += "_tv{:.2f}".format(joystick_policy.target_yaw_provider.get_target_velocity(joystick_policy.robot))
    
    if hasattr(joystick_policy.target_yaw_provider, "keep_target_delta_yaw_steps_avg") and hasattr(joystick_policy.target_yaw_provider, "keep_velocity_steps_avg"):
        setattr(joystick_policy.target_yaw_provider, "keep_target_delta_yaw_steps_avg", task_configs.joystick_keep_target_delta_yaw_steps_avg)
        setattr(joystick_policy.target_yaw_provider, "keep_velocity_steps_avg", task_configs.joystick_keep_target_velocity_steps_avg)
        task_suffix += "_keep,yaw{:d},tv{:d}".format(task_configs.joystick_keep_target_delta_yaw_steps_avg, task_configs.joystick_keep_target_velocity_steps_avg)

    if (
        hasattr(joystick_policy.reward_provider, "pitch_rate_penalty_factor") and 
        hasattr(joystick_policy.reward_provider, "roll_rate_penalty_factor")
    ):
        setattr(joystick_policy.reward_provider, "pitch_rate_penalty_factor", task_configs.rew_pitch_rate_penalty_factor)
        setattr(joystick_policy.reward_provider, "roll_rate_penalty_factor", task_configs.rew_roll_rate_penalty_factor)
        task_suffix += "_avpf{:.2f},{:.2f}".format(task_configs.rew_pitch_rate_penalty_factor, task_configs.rew_roll_rate_penalty_factor)
    
    if (
        hasattr(joystick_policy.reward_provider, "joint_diagonal_penalty_weight") and
        hasattr(joystick_policy.reward_provider, "joint_shoulder_penalty_weight")
    ):
        setattr(joystick_policy.reward_provider, "joint_diagonal_penalty_weight", task_configs.rew_joint_diagonal_penalty_weight)
        setattr(joystick_policy.reward_provider, "joint_shoulder_penalty_weight", task_configs.rew_joint_shoulder_penalty_weight)
        task_suffix += "_jspw{:.2e},{:.2e}".format(task_configs.rew_joint_diagonal_penalty_weight, task_configs.rew_joint_shoulder_penalty_weight)

    if (
        hasattr(joystick_policy.reward_provider, "joint_acc_penalty_weight") and
        hasattr(joystick_policy.reward_provider, "joint_vel_penalty_weight") and 
        hasattr(joystick_policy.reward_provider, "thigh_torque_penalty_weight")
    ):
        setattr(joystick_policy.reward_provider, "joint_acc_penalty_weight", task_configs.rew_joint_acc_penalty_weight)
        setattr(joystick_policy.reward_provider, "joint_vel_penalty_weight", task_configs.rew_joint_vel_penalty_weight)
        setattr(joystick_policy.reward_provider, "thigh_torque_penalty_weight", task_configs.rew_thigh_torque_penalty_weight)
        task_suffix += "_qpw{:.2e},{:.2e},{:.2e}".format(task_configs.rew_joint_acc_penalty_weight, task_configs.rew_joint_vel_penalty_weight, task_configs.rew_thigh_torque_penalty_weight)


    if hasattr(joystick_policy.reward_provider, "energy_penalty_weight"):
        setattr(joystick_policy.reward_provider, "energy_penalty_weight", task_configs.rew_energy_penalty_weight)
        task_suffix += "_ep{:.3f}".format(task_configs.rew_energy_penalty_weight) if task_configs.rew_energy_penalty_weight > 0.0 else "_nep"
    
    if hasattr(joystick_policy.reward_provider, "qpos_penalty_weight"):
        setattr(joystick_policy.reward_provider, "qpos_penalty_weight", task_configs.rew_qpos_penalty_weight)
        task_suffix += "_qpw{:.3f}".format(task_configs.rew_qpos_penalty_weight)

    if hasattr(joystick_policy.reward_provider, "smooth_torque_penalty_weight"):
        setattr(joystick_policy.reward_provider, "smooth_torque_penalty_weight", task_configs.rew_smooth_torque_penalty_weight)
        task_suffix += "_stpw{:.2e}".format(task_configs.rew_smooth_torque_penalty_weight)

    if hasattr(joystick_policy.reward_provider, "contact_reward_weight"):
        setattr(joystick_policy.reward_provider, "contact_reward_weight", task_configs.rew_contact_reward_weight)
        task_suffix += "_crw{:.2e}".format(task_configs.rew_contact_reward_weight)

    if task_configs.enable_reset_policy and not is_relabel_task:
        env = _helper_apply_resetter_wrapper(env, task_configs, reset_agent_config)
        task_suffix = "_resetter" + task_suffix
    
    if task_configs.limit_episode_length > 0 and not is_relabel_task:
        env = gym.wrappers.TimeLimit(
            env,
            max_episode_steps=task_configs.limit_episode_length
        )

    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
    env = envwrappers.JoystickPolicyTruncationWrapper(env)
    
    env, obs_space_before_flatten = common_post_process_wrapper(
        env, 
        action_range = task_configs.action_range, 
        filter_actions=task_configs.filter_actions, 
        frame_stack=task_configs.frame_stack, 
        action_history=task_configs.action_history,
        center_init_action=task_configs.center_init_action,
        log_print=True
    )

    return task_suffix, env, obs_space_before_flatten

def apply_task_configs(
    env: JoystickEnvironment,
    env_name: str,
    max_steps: int,
    task_configs,
    reset_agent_config,
    is_eval : bool = False
) -> Tuple[str, gym.Env]:
    suffix, wrapped_env, obs_space_before_flatten = _apply_task_configs(env, task_configs, reset_agent_config, False)
    if task_configs.relabel_count <= 0 or is_eval:
        return suffix, wrapped_env
    
    if not task_configs.symmetric_relabel:
        relabel_envs = []
        if env.joystick_policy.target_yaw_provider.is_target_velocity_fixed():
            fixed_v = env.joystick_policy.target_yaw_provider.get_target_velocity(env.joystick_policy.robot)
            target_velocity_lambda = lambda robot: fixed_v
            target_fixed_velocity_lambda = lambda: True
        else:
            #target_velocity_lambda = lambda robot: env.joystick_policy.target_yaw_provider.get_target_velocity(robot)
            target_velocity_lambda = None
            target_fixed_velocity_lambda = None

        readonly_robot = ReadOnlyWalkerWrapper(
            env.joystick_policy.robot
        )

        for i in range(task_configs.relabel_count):
            relabel_joystick_task = JoystickPolicy(
                readonly_robot,
                copy.copy(env.joystick_policy.reward_provider),
                RelabelTargetProvider(env.joystick_policy.target_yaw_provider),
                [],
                [],
                [],
                [],
                copy.copy(env.joystick_policy.target_observable),
                env.joystick_policy.enabled_observables
            )
            relabel_env = JoystickEnvImpl(relabel_joystick_task)
            if target_velocity_lambda is not None:
                relabel_joystick_task.target_yaw_provider.get_target_velocity = target_velocity_lambda
            if target_fixed_velocity_lambda is not None:
                relabel_joystick_task.target_yaw_provider.is_target_velocity_fixed = target_fixed_velocity_lambda
            relabel_envs.append(_apply_task_configs(relabel_env, task_configs, reset_agent_config, True)[1])
        
        suffix = "_rlab{}".format(task_configs.relabel_count) + suffix

        ret_env = RelabelAggregateWrapper(
            wrapped_env, max_steps, relabel_envs
        )
    else:
        suffix = "_rlabsym" + suffix
        task_configs.relabel_update_fraction = 0.5 # Force to 0.5
        task_configs.relabel_count = 1 # Force to 1
        ret_env = SymmetricRelabelWrapper(
            wrapped_env, obs_space_before_flatten, max_steps
        )

    return suffix, ret_env
    