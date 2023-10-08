import numpy as np
from rail_walker_interface import JoystickPolicyTerminationConditionProvider
from rail_walker_interface import BaseWalker
from typing import Any, Optional


class JoystickPolicyRollPitchTerminationConditionProvider(JoystickPolicyTerminationConditionProvider[BaseWalker]):
    def __init__(self, terminate_roll_rad : float = 60.0 / 180.0 * np.pi, terminate_pitch_min_rad = -60.0 / 180.0 * np.pi, terminate_pitch_max_rad = 60.0/180.0 * np.pi, reset_fall_count : bool = False) -> None:
        super().__init__()
        self.terminate_roll_rad = terminate_roll_rad
        self.terminate_pitch_min_rad = terminate_pitch_min_rad
        self.terminate_pitch_max_rad = terminate_pitch_max_rad
        self._should_term = False
        self.reset_fall_count = reset_fall_count
        self.fall_count = 0
    
    def should_terminate(self) -> bool:
        return self._should_term

    def add_fall_count_info(self, info_dict: dict[str,Any]) -> None:
        self.fall_count += 1
        info_dict["fall_count"] = self.fall_count

    def step_termination_condition(
        self, 
        Robot: BaseWalker, 
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
        roll, pitch, _ = Robot.get_roll_pitch_yaw()
        if abs(roll) > self.terminate_roll_rad or pitch < self.terminate_pitch_min_rad or pitch > self.terminate_pitch_max_rad:
            self._should_term = True
            self.add_fall_count_info(info_dict)
        else:
            self._should_term = False

    def reset_termination_condition(
        self, 
        Robot: BaseWalker, 
        info_dict: dict[str,Any], 
        termination_provider_triggered,
        randomState: np.random.RandomState
    ) -> None:
        self._should_term = False
        if self.reset_fall_count:
            self.fall_count = 0
        if "fall_count" not in info_dict:
            info_dict["fall_count"] = self.fall_count

class JoystickPolicyResetterEarlyStopTruncationProvider(JoystickPolicyTerminationConditionProvider[BaseWalker]):
    def __init__(
            self, 
            terminate_roll_pitch_rad : float = 10.0 / 180.0 * np.pi, 
            terminate_qpos_range : float = 0.3,
            terminate_qvel_range : float = 0.02
        ) -> None:
        super().__init__()
        self.terminate_roll_pitch_rad = terminate_roll_pitch_rad
        self.terminate_qpos_range = terminate_qpos_range
        self.terminate_qvel_range = terminate_qvel_range
        self._should_term = False
    
    def should_terminate(self) -> bool:
        return self._should_term

    def step_termination_condition(
        self, 
        Robot: BaseWalker, 
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
        roll, pitch, yaw = Robot.get_roll_pitch_yaw()
        qpos = Robot.get_joint_qpos()
        qvel = Robot.get_joint_qvel()

        dqpos = qpos - Robot.joint_qpos_init
        dqpos_maxrange = Robot.joint_qpos_maxs - Robot.joint_qpos_mins

        rp_terminate = abs(roll) < self.terminate_roll_pitch_rad or abs(pitch) < self.terminate_roll_pitch_rad
        qpos_terminate = np.all(dqpos < dqpos_maxrange * self.terminate_qpos_range)
        qvel_terminate = np.all(abs(qvel) < self.terminate_qvel_range)
        self._should_term = rp_terminate and qpos_terminate and qvel_terminate
    
    def reset_termination_condition(
        self, 
        Robot: BaseWalker, 
        info_dict: dict[str,Any], 
        termination_provider_triggered,
        randomState: np.random.RandomState
    ) -> None:
        self._should_term = False
