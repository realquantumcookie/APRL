from rail_real_walker.robots.go1 import Go1RealWalker
from rail_walker_interface import JoystickPolicyTargetProvider, JoystickPolicyTerminationConditionProvider
from unitree_go1_wrapper import XRockerBtnDataStruct
import numpy as np
from typing import Any, Optional
import gym
import gym.spaces

def normalize_rad(rad : float) -> float:
    return ((rad + np.pi) % (2 * np.pi)) - np.pi

"""
Real Joystick Target Provider using the Go1's controller
"""
class JoystickPolicyGo1JoystickTargetProvider(JoystickPolicyTargetProvider[Go1RealWalker]):
    def __init__(
        self,
        ahead_distance : float = 1.0,
        min_target_velocity : float = 0.0,
        max_target_velocity : float = 1.0,
        max_velocity_change_rate : float = 0.2,
        max_target_yaw_change_rate : float = 5.0 / 180.0 * np.pi,
        min_hold_steps : int = 20 * 1,
        max_hold_steps : int = 20 * 3,
        target_merge_to_zero_interval : float = 10 / 180 * np.pi,
    ):
        self.ahead_distance = ahead_distance
        self.min_target_velocity = min_target_velocity
        self.max_target_velocity = max_target_velocity
        self.max_velocity_change_rate = max_velocity_change_rate
        self.max_tdy_change_rate = max_target_yaw_change_rate
        self.min_hold_steps = min_hold_steps
        self.max_hold_steps = max_hold_steps
        self.target_merge_to_zero_interval = target_merge_to_zero_interval

        self._tagret_delta_yaw = 0.0
        self._new_target_delta_yaw= 0.0
        self._target_velocity = (min_target_velocity + max_target_velocity) / 2
        self._new_target_velocity = self._target_velocity
        self._remaining_hold_min_steps = min_hold_steps
        self._remaining_hold_max_steps = max_hold_steps
    
    def get_target_goal_world_delta(self, Robot: Go1RealWalker) -> np.ndarray:
        return self.get_world_delta(Robot, self._target_delta_yaw)

    def get_target_velocity(self, Robot: Go1RealWalker) -> float:
        return self._target_velocity
    
    def is_target_velocity_fixed(self) -> bool:
        return False

    def step_target(
        self, 
        Robot: Go1RealWalker, 
        info_dict: dict[str,Any], 
        randomState : np.random.RandomState
    ) -> None:
        roll, pitch, yaw = Robot.get_roll_pitch_yaw()
        real_tdy, real_target_v = self.get_joystick_real_target(Robot, yaw)
        if np.abs(real_tdy) < self.target_merge_to_zero_interval / 2.0:
            real_tdy = 0.0
        self._new_target_delta_yaw = real_tdy
        self._new_target_velocity = real_target_v
        self._remaining_hold_max_steps -= 1
        self._remaining_hold_min_steps -= 1
    
    def get_joystick_real_target(self, Robot : Go1RealWalker, robot_yaw : float) -> tuple[float, float]:
        joystick_raw = Robot.get_joystick_values()[0]
        
        if np.all(joystick_raw <= 0.05):
            return 0.0, 0.0

        local_target_yaw = np.arctan2(-joystick_raw[0], joystick_raw[1])
        real_target_v = np.sqrt(joystick_raw[0]**2 + joystick_raw[1]**2) / np.sqrt(2) * self.max_target_velocity
        return local_target_yaw, real_target_v

    def has_target_changed(self) -> bool:
        return self._remaining_hold_max_steps <= 0 or ((
            np.abs(normalize_rad(self._new_target_delta_yaw - self._target_delta_yaw)) > self.max_tdy_change_rate or
            np.abs(self._new_target_velocity - self._target_velocity) > self.max_velocity_change_rate
        ) and self._remaining_hold_min_steps <= 0)

    def after_step_target(
        self, 
        Robot: Go1RealWalker, 
        info_dict: dict[str,Any], 
        randomState : np.random.RandomState
    ) -> None:
        self._remaining_hold_min_steps = self.min_hold_steps
        self._remaining_hold_max_steps = self.max_hold_steps
        self._target_delta_yaw = self._new_target_delta_yaw
        self._target_velocity = self._new_target_velocity
        return super().after_step(Robot, info_dict, randomState)

    def reset_target(
        self, 
        Robot: Go1RealWalker, 
        info_dict: dict[str,Any], 
        termination_provider_triggered: JoystickPolicyTerminationConditionProvider,
        randomState: np.random.RandomState
    ) -> None:
        self._remaining_hold_max_steps = self.max_hold_steps
        self._remaining_hold_min_steps = self.min_hold_steps

        # Update joystick target
        self._target_delta_yaw, self._target_velocity = self.get_joystick_real_target(Robot, Robot.get_roll_pitch_yaw()[2])
        if np.abs(self._target_delta_yaw) < self.target_merge_to_zero_interval / 2.0:
            self._target_delta_yaw = 0.0
        
        self._new_target_delta_yaw = self._target_delta_yaw
        self._new_target_velocity = self._target_velocity

    def get_world_delta(self, robot: Go1RealWalker, tdy : float) -> np.ndarray:
        current_yaw = robot.get_roll_pitch_yaw()[2]
        target_yaw = normalize_rad(tdy + current_yaw)
        return np.array([np.cos(target_yaw),np.sin(target_yaw)]) * self.ahead_distance

class JoystickPolicyGo1JoystickSimpleTargetProvider(JoystickPolicyGo1JoystickTargetProvider):
    def __init__(
        self, 
        ahead_distance: float = 1, 
        min_target_velocity: float = 0, 
        max_target_velocity: float = 1, 
        max_velocity_change_rate: float = 0.2, 
        min_hold_steps: int = 20 * 1, 
        max_hold_steps: int = 20 * 3
    ):
        super().__init__(
            ahead_distance, 
            min_target_velocity, 
            max_target_velocity, 
            max_velocity_change_rate, 
            40/180*np.pi, 
            min_hold_steps, 
            max_hold_steps, 
            np.pi / 2
        )
    
    def step_target(self, Robot: Go1RealWalker, info_dict: dict[str, Any], randomState: np.random.RandomState) -> None:
        super().step_target(Robot, info_dict, randomState)
        sign_ntdy = np.sign(self._new_target_delta_yaw)
        self._new_target_delta_yaw = sign_ntdy * np.pi / 2
        self._new_target_velocity = 0.5 if sign_ntdy == 0 else 0.3
        #self._new_target_velocity = 0.8

    def reset_target(
        self, 
        Robot: Go1RealWalker, 
        info_dict: dict[str, Any], 
        termination_provider_triggered: JoystickPolicyTerminationConditionProvider, 
        randomState: np.random.RandomState
    ) -> None:
        super().reset_target(Robot, info_dict, termination_provider_triggered, randomState)
        sign_ntdy = np.sign(self._new_target_delta_yaw)
        self._new_target_delta_yaw = sign_ntdy * np.pi / 2
        self._new_target_velocity = 0.5 if sign_ntdy == 0 else 0.3
        #self._new_target_velocity = 0.8
        self._target_delta_yaw = self._new_target_delta_yaw
        self._target_velocity = self._new_target_velocity