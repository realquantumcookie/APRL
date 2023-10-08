import gym
import gym.spaces
import numpy as np
from rail_walker_interface import JoystickPolicyTargetProvider, JoystickPolicyTerminationConditionProvider
from rail_walker_interface import BaseWalker
from typing import Any, Optional
from rail_walker_interface.joystick_policy.joystick_interfaces import JoystickPolicyTerminationConditionProvider

from rail_walker_interface.robot.robot import BaseWalker

def normalize_rad(rad : float) -> float:
    return (rad + np.pi) % (2 * np.pi) - np.pi

class JoystickPolicyForwardOnlyTargetProvider(JoystickPolicyTargetProvider[BaseWalker]):
    def __init__(self, ahead_distance : float = 1.0, target_linear_velocity : float = 1.0) -> None:
        super().__init__()
        self.ahead_distance = ahead_distance
        self.target_linear_velocity = target_linear_velocity
        self._world_delta = np.zeros(2)

    def get_target_goal_world_delta(self, Robot: BaseWalker) -> np.ndarray:
        return self._world_delta
    
    def get_target_velocity(self, Robot: BaseWalker) -> float:
        return self.target_linear_velocity
    
    def is_target_velocity_fixed(self) -> bool:
        return True

    def has_target_changed(self) -> bool:
        return False

    def step_target(
        self, 
        Robot: BaseWalker, 
        info_dict: dict[str,Any], 
        randomState : np.random.RandomState
    ) -> None:
        _, _, yaw = Robot.get_roll_pitch_yaw()
        self._update_target(yaw)

    def reset_target(
        self, 
        Robot: BaseWalker, 
        info_dict: dict[str,Any], 
        termination_provider_triggered: JoystickPolicyTerminationConditionProvider,
        randomState: np.random.RandomState
    ) -> None:
        _, _, yaw = Robot.get_roll_pitch_yaw()
        self._update_target(yaw)

    def _update_target(self, yaw : float):
        self._world_delta[0] = np.cos(yaw) * self.ahead_distance
        self._world_delta[1] = np.sin(yaw) * self.ahead_distance

"""
This class mimics a behavior of a joystick, it follows a random walk time series and, 
with a small chance each time, randomly jumps to a sample from a uniform distribution in the range [-pi, pi)
"""
class JoystickPolicyAutoJoystickTargetProvider(JoystickPolicyTargetProvider[BaseWalker]):
    def __init__(
        self,
        random_walk_std_per_step : list[tuple[float, float]]= [
            (0.6, 30 / 180 * np.pi), # with 0.8 probability, sample from a normal distribution with std 30 degrees
            (0.4, 90 / 180 * np.pi), # with 0.2 probability, sample from a normal distribution with std 90 degrees
        ],
        random_jump_prob : float = 0.4,
        ahead_distance : float = 1.0,
        min_target_velocity : float = 0.0,
        max_target_velocity : float = 1.0,
        max_velocity_change_rate : float = 0.3,
        velocity_probability_distribution : Optional[
            list[tuple[float, float]]
        ] = [
            (0.1, 0.05),
            (0.1, 0.3),
            (0.5, 0.7),
            (0.1, 0.9),
            (0.2, 1.0)
        ],
        target_merge_to_zero_interval : float = 10 / 180 * np.pi,
        keep_target_delta_yaw_steps_avg : int = 20 * 3,
        keep_velocity_steps_avg : int = 20 * 3,
    ):
        self.random_walk_std_per_step= random_walk_std_per_step
        self.random_jump_prob= random_jump_prob
        self.ahead_distance = ahead_distance
        self.min_target_velocity = min_target_velocity
        self.max_target_velocity = max_target_velocity
        self.max_velocity_change_rate = max_velocity_change_rate
        self.velocity_probability_distribution = velocity_probability_distribution
        self.target_merge_to_zero_interval = target_merge_to_zero_interval
        self.keep_target_delta_yaw_steps_avg = keep_target_delta_yaw_steps_avg
        self.keep_velocity_steps_avg = keep_velocity_steps_avg

        self._target_delta_yaw = 0.0
        self._target_velocity = (min_target_velocity + max_target_velocity) / 2
        self._steps_until_next_target_velocity_change = 0
        self._steps_until_next_target_delta_yaw_change = 0
        self._should_change_velocity = False
        self._should_change_delta_yaw = False
    
    def get_target_goal_world_delta(self, Robot: BaseWalker) -> np.ndarray:
        target_yaw = self._target_delta_yaw + Robot.get_roll_pitch_yaw()[2]
        return np.array([
            np.cos(target_yaw), np.sin(target_yaw)
        ]) * self.ahead_distance
    
    def get_target_velocity(self, Robot: BaseWalker) -> float:
        return self._target_velocity
    
    def is_target_velocity_fixed(self) -> bool:
        return False

    def has_target_changed(self) -> bool:
        return self._should_change_velocity or self._should_change_delta_yaw

    def step_target(
        self, 
        Robot: BaseWalker, 
        info_dict: dict[str,Any], 
        randomState : np.random.RandomState
    ) -> None:
        if self._steps_until_next_target_delta_yaw_change > 1:
            self._steps_until_next_target_delta_yaw_change -= 1
            self._should_change_delta_yaw = False
        else:
            self._should_change_delta_yaw = True
        
        if self._steps_until_next_target_velocity_change > 1:
            self._steps_until_next_target_velocity_change -= 1
            self._should_change_velocity = False
        else:
            self._should_change_velocity = True
    
    def after_step_target(
        self, 
        Robot: BaseWalker, 
        info_dict: dict[str,Any], 
        randomState : np.random.RandomState
    ) -> None:
        if self._should_change_delta_yaw:
            self._steps_until_next_target_delta_yaw_change = randomState.poisson(self.keep_target_delta_yaw_steps_avg) if self.keep_target_delta_yaw_steps_avg > 0 else 0
            if randomState.uniform() < self.random_jump_prob:
                self._target_delta_yaw = randomState.uniform(-np.pi, np.pi)
            else:
                rnum = randomState.uniform()
                t_prob = 0
                t_std = 0.5 / 180 * np.pi # default to 0.5 degrees
                for prob, std in self.random_walk_std_per_step:
                    t_prob += prob
                    if rnum < t_prob:
                        t_std = std
                        break
                
                self._target_delta_yaw += randomState.normal(0, t_std)
                self._target_delta_yaw = normalize_rad(self._target_delta_yaw)
            if np.abs(self._target_delta_yaw) < self.target_merge_to_zero_interval / 2:
                self._target_delta_yaw = 0.0
            
            self._should_change_delta_yaw = False
        if self._should_change_velocity:
            self._steps_until_next_target_velocity_change = randomState.poisson(self.keep_velocity_steps_avg) if self.keep_velocity_steps_avg > 0 else 0
            rand_change_in_v_unit = randomState.uniform()
            if self.velocity_probability_distribution is None:
                change_in_velocity = rand_change_in_v_unit * (self.max_velocity_change_rate * 2) - self.max_velocity_change_rate
            else:
                probability_le_current_velocity = 0.0
                for i in range(len(self.velocity_probability_distribution)):
                    p, v = self.velocity_probability_distribution[i]
                    if v < self._target_velocity:
                        probability_le_current_velocity += p
                    else:
                        prev_p, prev_v = self.velocity_probability_distribution[i-1] if i > 0 else (0.0, self.min_target_velocity)
                        probability_le_current_velocity += (self._target_velocity - prev_v) / (v - prev_v) * p
                        break
                if rand_change_in_v_unit <= probability_le_current_velocity:
                    change_in_velocity = (rand_change_in_v_unit / probability_le_current_velocity - 1.0) * self.max_velocity_change_rate
                else:
                    probability_g_current_velocity = 1.0 - probability_le_current_velocity
                    change_in_velocity = (rand_change_in_v_unit - probability_le_current_velocity) / probability_g_current_velocity * self.max_velocity_change_rate
            
            self._target_velocity += change_in_velocity
            self._target_velocity = np.clip(self._target_velocity, self.min_target_velocity, self.max_target_velocity)
            self._should_change_velocity = False

    def reset_target(
        self, 
        Robot: BaseWalker, 
        info_dict: dict[str,Any], 
        termination_provider_triggered: JoystickPolicyTerminationConditionProvider,
        randomState: np.random.RandomState
    ) -> None:
        self._target_delta_yaw = randomState.uniform(-np.pi, np.pi)
        if np.abs(self._target_delta_yaw) < self.target_merge_to_zero_interval / 2:
            self._target_delta_yaw = 0.0
        rand_v = randomState.uniform()
        if self.velocity_probability_distribution is None:
            self._target_velocity = rand_v * (self.max_target_velocity - self.min_target_velocity) + self.min_target_velocity
        else:
            accumulated_probability = 0.0
            for i in range(len(self.velocity_probability_distribution)):
                p,v = self.velocity_probability_distribution[i]
                if rand_v > accumulated_probability + p:
                    accumulated_probability += p
                else:
                    prev_p, prev_v = self.velocity_probability_distribution[i-1] if i > 0 else (0.0, self.min_target_velocity)
                    self._target_velocity = prev_v + (rand_v - accumulated_probability) / p * (v - prev_v)
                    break

        self._steps_until_next_target_velocity_change = randomState.poisson(self.keep_velocity_steps_avg) if self.keep_velocity_steps_avg > 0 else 0
        self._steps_until_next_target_delta_yaw_change = randomState.poisson(self.keep_target_delta_yaw_steps_avg) if self.keep_target_delta_yaw_steps_avg > 0 else 0
        self._should_change_velocity = False
        self._should_change_delta_yaw = False

class JoystickPolicyAutoJoystickSimpleTargetProvider(JoystickPolicyTargetProvider[BaseWalker]):
    def __init__(
        self,
        ahead_distance : float = 1.0,
        avg_num_steps_per_target : int = 20 * 2,
        target_linear_velocity : float = 0.5
    ) -> None:
        super().__init__()
        self.ahead_distance = ahead_distance
        self.avg_num_steps_per_target = avg_num_steps_per_target
        self.target_linear_velocity = target_linear_velocity
        self._target_delta_yaw = 0.0
        self._next_target_delta_yaw = 0.0
        self._steps_until_next_target = 0

    def get_world_delta(self, target_yaw : float) -> np.ndarray:
        return np.array([np.cos(target_yaw), np.sin(target_yaw)]) * self.ahead_distance
    
    def get_target_velocity(self, Robot: BaseWalker) -> float:
        return self.target_linear_velocity
    
    def is_target_velocity_fixed(self) -> bool:
        return True

    def get_target_goal_world_delta(self, Robot: BaseWalker) -> np.ndarray:
        return self.get_world_delta(self._target_delta_yaw + Robot.get_roll_pitch_yaw()[2])
    
    def has_target_changed(self) -> bool:
        return self._target_delta_yaw != self._next_target_delta_yaw
    
    def step_target(self, Robot: BaseWalker, info_dict: dict[str, Any], randomState: np.random.RandomState) -> None:
        if self._steps_until_next_target >= 1:
            self._steps_until_next_target -= 1
        else:
            self._steps_until_next_target = randomState.poisson(self.avg_num_steps_per_target)
            self._next_target_delta_yaw = randomState.choice(
                np.array([-np.pi/2.0, 0.0, np.pi/2.0])
            )
    
    def after_step_target(self, Robot: BaseWalker, info_dict: dict[str, Any], randomState: np.random.RandomState) -> None:
        self._target_delta_yaw = self._next_target_delta_yaw
    
    def reset_target(
        self, 
        Robot: BaseWalker, 
        info_dict: dict[str, Any], 
        termination_provider_triggered: JoystickPolicyTerminationConditionProvider, 
        randomState: np.random.RandomState
    ) -> None:
        self._next_target_delta_yaw = randomState.choice(
            np.array([-np.pi/2.0, 0.0, np.pi/2.0])
        )
        self._target_delta_yaw = self._next_target_delta_yaw
        self._steps_until_next_target = randomState.poisson(self.avg_num_steps_per_target)

class JoystickPolicyAutoJoystickSimplePlusTargetProvider(JoystickPolicyTargetProvider[BaseWalker]):
    def __init__(
        self,
        ahead_distance : float = 1.0,
        avg_num_steps_per_target : int = 20 * 2,
        min_target_velocity : float = 0.2,
        max_target_velocity : float = 1.0,
    ) -> None:
        super().__init__()
        self.ahead_distance = ahead_distance
        self.avg_num_steps_per_target = avg_num_steps_per_target
        self.min_target_velocity = min_target_velocity
        self.max_target_velocity = max_target_velocity

        self._target_delta_yaw = 0.0
        self._target_velocity = 0.0
        self._next_target_delta_yaw = 0.0
        self._next_target_velocity = 0.0
        self._steps_until_next_target = 0
        
    
    @property
    def current_custom_target_velocity(self) -> float:
        return self._target_velocity
    
    def get_world_delta(self, target_yaw : float) -> np.ndarray:
        return np.array([np.cos(target_yaw), np.sin(target_yaw)]) * self.ahead_distance
    
    def get_target_velocity(self, Robot: BaseWalker) -> float:
        return self._target_velocity
    
    def is_target_velocity_fixed(self) -> bool:
        return False

    def get_target_goal_world_delta(self, Robot: BaseWalker) -> np.ndarray:
        return self.get_world_delta(self._target_delta_yaw + Robot.get_roll_pitch_yaw()[2])
    
    def has_target_changed(self) -> bool:
        return self._target_delta_yaw != self._next_target_delta_yaw or self._target_velocity != self._next_target_velocity
    
    def step_target(self, Robot: BaseWalker, info_dict: dict[str, Any], randomState: np.random.RandomState) -> None:
        if self._steps_until_next_target >= 1:
            self._steps_until_next_target -= 1
        else:
            self._steps_until_next_target = randomState.poisson(self.avg_num_steps_per_target)
            self._next_target_delta_yaw = randomState.choice(
                np.array([-np.pi/2.0, 0.0, np.pi/2.0])
            )
            self._next_target_velocity = randomState.uniform(self.min_target_velocity, self.max_target_velocity)
    
    def after_step_target(self, Robot: BaseWalker, info_dict: dict[str, Any], randomState: np.random.RandomState) -> None:
        self._target_delta_yaw = self._next_target_delta_yaw
        self._target_velocity = self._next_target_velocity
    
    def reset_target(
        self, 
        Robot: BaseWalker, 
        info_dict: dict[str, Any], 
        termination_provider_triggered: JoystickPolicyTerminationConditionProvider, 
        randomState: np.random.RandomState
    ) -> None:
        self._next_target_delta_yaw = randomState.choice(
            np.array([-np.pi/2.0, 0.0, np.pi/2.0])
        )
        self._next_target_velocity = randomState.uniform(self.min_target_velocity, self.max_target_velocity)
        self._target_delta_yaw = self._next_target_delta_yaw
        self._target_velocity = self._next_target_velocity
        self._steps_until_next_target = randomState.poisson(self.avg_num_steps_per_target)
    