import ml_collections
from ml_collections.config_dict import config_dict


def get_config():
    config = ml_collections.ConfigDict()

    # ====================== Shared Control Configs ======================
    config.Kp = 60.0
    config.Kd = 5.0
    config.action_interpolation = False
    config.filter_actions = 0
    config.action_range = 1.0
    config.center_init_action = False
    config.power_protect_factor = 0.6

    # ====================== Environment Configs ======================
    config.frame_stack = 3
    config.action_history = 3
    config.limit_episode_length = 0 # 0 means no limit

    # ====================== Joystick Policy Configs ======================
    config.lock_target = False
    config.rew_target_velocity = 0.5
    config.rew_smooth_change_in_tdy_steps = 5
    config.rew_qpos_penalty_weight = 0.0
    config.rew_energy_penalty_weight = 0.0
    config.rew_smooth_torque_penalty_weight = 0.0
    config.rew_joint_shoulder_penalty_weight = 0.0
    config.rew_joint_diagonal_penalty_weight = 0.0
    config.rew_joint_acc_penalty_weight = 0.0
    config.rew_joint_vel_penalty_weight = 0.0
    config.rew_pitch_rate_penalty_factor = 0.0
    config.rew_roll_rate_penalty_factor = 0.0
    config.rew_thigh_torque_penalty_weight = 0.0
    config.rew_contact_reward_weight = 0.0

    # ====================== Joystick Target Configs ======================
    config.min_target_velocity = 0.3
    config.max_target_velocity = 1.0
    config.joystick_keep_target_delta_yaw_steps_avg = 20 * 15
    config.joystick_keep_target_velocity_steps_avg = 20 * 15
    config.max_velocity_change_rate = 0.3

    # ====================== Reset Wrapper Configs ======================
    config.enable_reset_policy = False
    config.reset_policy_joystick_enabled_observables = ["joints_pos", "imu"]
    config.reset_policy_max_seconds = 5.0
    config.reset_policy_Kp = 40.0
    config.reset_policy_Kd = 5.0
    config.reset_policy_action_interpolation = True
    config.reset_policy_power_protect_factor = 0.7
    config.reset_policy_filter_actions = 0
    config.reset_policy_action_range = 1.0
    config.reset_policy_frame_stack = 3
    config.reset_policy_action_history = 3
    config.reset_policy_center_init_action = False

    config.reset_agent_checkpoint = "./reset_ckpts/Go1_PD40.00,5.00_ai_ar1.00_fs3_ppf0.80_ep_new"

    # ====================== Relabel Wrapper Configs ======================
    config.relabel_count = 0
    config.symmetric_relabel = False
    config.relabel_update_fraction = 0.5

    return config