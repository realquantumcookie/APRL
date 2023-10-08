from rail_walker_interface import JoystickPolicyRewardProvider, JoystickPolicyTerminationConditionProvider
from rail_mujoco_walker import JoystickPolicyProviderWithDMControl, add_sphere_to_mjv_scene, add_arrow_to_mjv_scene, JoystickPolicyDMControlTask, RailSimWalkerDMControl, find_dm_control_non_contacting_height
from dm_control.mujoco.engine import Physics as EnginePhysics
import mujoco
import numpy as np
from typing import Any, Tuple, Optional
import transforms3d as tr3d

JOINT_WEIGHTS = np.array([1.0, 0.75, 0.5] * 4)

class ResetRewardProvider(JoystickPolicyRewardProvider[RailSimWalkerDMControl],JoystickPolicyProviderWithDMControl):
    SUCCESS_RATE_REW_THRES = 0.8

    def __init__(
            self,
            use_energy_penalty: bool = False,
        ) -> None:
        JoystickPolicyRewardProvider.__init__(self)
        JoystickPolicyProviderWithDMControl.__init__(self)
        self.use_energy_penalty = use_energy_penalty
        self.rew = 0.0
        self.standup_height : float = None
    
    def get_reward(self) -> float:
        return self.rew

    def reset_reward(
        self, 
        Robot: RailSimWalkerDMControl, 
        info_dict: dict[str,Any], 
        termination_provider_triggered: JoystickPolicyTerminationConditionProvider,
        randomState: np.random.RandomState
    ) -> None:
        self.rew = 0.0
        info_dict["reward_roll"] = 0.0
        info_dict["reward_height"] = 0.0
        info_dict["reward_pose"] = 0.0
        info_dict["reward_qvel"] = 0.0
        info_dict["reward_stand"] = 0.0
        info_dict["cos_dist"] = 0.0
        info_dict["energy"] = 0.0
        info_dict["success_rate"] = 0.0
        self.standup_height = 0.325

        # # Retrieve the target height
        # physics = Robot.mujoco_walker._last_physics
        # # Remember the previous state of the robot
        # prev_joint = Robot.get_joint_qpos()
        # prev_pos, prev_quat = Robot.mujoco_walker.get_pose(physics)
        # # Set the robot to standup pose
        # find_dm_control_non_contacting_height(physics, Robot.mujoco_walker, prev_pos[0], prev_pos[1], Robot.joint_qpos_init, prev_quat)
        # self.standup_height = Robot.mujoco_walker.get_position(physics)[2] # retrieve the height of the robot
        # # reset the robot back to previous state
        # with physics.reset_context():
        #     Robot.mujoco_walker.set_pose(physics, prev_pos, prev_quat)
        #     physics.bind(Robot.mujoco_walker.joints).qpos[:] = prev_joint
        # #print("Standup height: ", self.standup_height)

    def _log_success_rate(
        self, 
        info_dict: dict[str,Any], 
        rew_roll : float,
        rew_height : float,
        rew_pose : float
    ) -> None:
        if rew_roll > __class__.SUCCESS_RATE_REW_THRES and rew_height > __class__.SUCCESS_RATE_REW_THRES and rew_pose > __class__.SUCCESS_RATE_REW_THRES:
            info_dict["success_rate"] = 1.0
        else:
            info_dict["success_rate"] = 0.0

    def _calc_reward_roll(
        self,
        Robot: RailSimWalkerDMControl
    ) -> Tuple[float, float]:
        cos_dist = tr3d.quaternions.quat2mat(Robot.get_framequat_wijk())[-1,-1] # See if the z-axis is pointing up
        r_roll = (0.5 * cos_dist + 0.5)**2
        return r_roll, cos_dist

    def _calc_reward_stand(
        self,
        Robot: RailSimWalkerDMControl
    ) -> float:
        # height match reward
        tar_h = self.standup_height
        # root_h = Robot.get_root_height()
        root_h = Robot.get_3d_location()[-1]
        # root_h = physics.bind(self._robot.root_body).xpos[-1]
        h_err = tar_h - root_h
        h_err /= tar_h
        h_err = np.clip(h_err, 0.0, 1.0)
        r_height = 1.0 - h_err

        # pose match reward
        joint_pose = Robot.get_joint_qpos()
        tar_pose = Robot.joint_qpos_init
        pose_diff = tar_pose - joint_pose
        pose_diff = JOINT_WEIGHTS * JOINT_WEIGHTS * pose_diff * pose_diff
        pose_err = np.sum(pose_diff)
        r_pose = np.exp(-0.6 * pose_err)

        # pose velocity reward
        tar_vel = 0.0
        joint_vel = Robot.get_joint_qvel()
        vel_diff = tar_vel - joint_vel
        vel_diff = vel_diff * vel_diff
        vel_err = np.sum(vel_diff)
        r_vel = np.exp(-0.02 * vel_err)

        r_stand = 0.2 * r_height + 0.6 * r_pose + 0.2 * r_vel

        return r_stand, r_height, r_pose, r_vel
    
    def render_scene_callback(self, task : JoystickPolicyDMControlTask, physics : EnginePhysics, scene : mujoco.MjvScene) -> None:
        robot_loc = task.robot.get_3d_location()
        sphere_loc = robot_loc.copy()
        sphere_loc[2] = self.standup_height
        #add_sphere_to_mjv_scene(scene, sphere_loc, 0.1, np.array([1.0, 0.0, 0.0, 1.0]))
        add_arrow_to_mjv_scene(scene, robot_loc, sphere_loc, 0.01, np.array([1.0, 0.0, 0.0, 1.0]))

        
    def step_reward(
        self, 
        Robot: RailSimWalkerDMControl, 
        action_target_qpos: np.ndarray,
        target_goal_world_delta: np.ndarray,
        target_goal_local: np.ndarray,
        target_yaw : float,
        target_delta_yaw: float, 
        target_velocity: float,
        velocity_to_goal: float, 
        change_in_abs_target_delta_yaw : float, 
        target_custom_data: Optional[Any],
        enable_target_custom_obs : bool,
        info_dict: dict[str,Any],
        randomState: np.random.RandomState
    ) -> None:
        roll_w = stand_w = 0.5
        roll_threshold = np.cos(0.2 * np.pi)

        r_roll, root_cos_dist = self._calc_reward_roll(Robot)
        r_stand, r_height, r_pose, r_vel = self._calc_reward_stand(Robot)

        info_dict["reward_roll"] = r_roll
        info_dict["reward_height"] = r_height
        info_dict["reward_pose"] = r_pose
        info_dict["reward_qvel"] = r_vel
        info_dict["cos_dist"] = root_cos_dist

        if root_cos_dist > roll_threshold:
            r_stand = r_stand
        else:
            r_stand = 0.0

        info_dict["reward_stand"] = r_stand

        reward = roll_w * r_roll + stand_w * r_stand
        standing_reward = reward

        #  Calculate Energy / Qvel Penalty
        qvel = Robot.get_joint_qvel()
        torque = Robot.get_joint_torques()
        energy = np.sum(np.abs(qvel * torque))
        info_dict["energy"] = energy
        
        if self.use_energy_penalty:
            energy_reward = - 0.02 * energy
        else:
            energy_reward = 0.0
        
        self._log_success_rate(
            info_dict,
            r_roll,
            r_height,
            r_pose
        )

        self.rew = standing_reward + energy_reward


class GatedResetRewardProvider(ResetRewardProvider):
    def step_reward(
        self, 
        Robot: RailSimWalkerDMControl,
        action_target_qpos: np.ndarray, 
        target_goal_world_delta: np.ndarray,
        target_goal_local: np.ndarray,
        target_yaw : float,
        target_delta_yaw: float, 
        target_velocity: float,
        velocity_to_goal: float, 
        change_in_abs_target_delta_yaw : float, 
        target_custom_data: Optional[Any],
        enable_target_custom_obs : bool,
        info_dict: dict[str,Any],
        randomState: np.random.RandomState
    ) -> None:

        r_roll, root_cos_dist = self._calc_reward_roll(Robot)
        info_dict["reward_roll"] = r_roll
        info_dict["cos_dist"] = root_cos_dist

        r_stand, r_height, r_pose, r_vel = self._calc_reward_stand(Robot)

        info_dict["reward_roll"] = r_roll
        info_dict["reward_height"] = r_height
        info_dict["reward_pose"] = r_pose
        info_dict["reward_qvel"] = r_vel
        info_dict["cos_dist"] = root_cos_dist

        #  Calculate Energy / Qvel Penalty
        qvel = Robot.get_joint_qvel()
        torque = Robot.get_joint_torques()
        energy = np.sum(np.abs(qvel * torque))
        info_dict["energy"] = energy
        
        if self.use_energy_penalty:
            energy_reward = - 0.02 * energy
        else:
            energy_reward = 0.0

        self._log_success_rate(
            info_dict,
            r_roll,
            r_height,
            r_pose
        )

        self.rew = (r_roll * (1 + r_stand) + energy_reward) / 2.0
