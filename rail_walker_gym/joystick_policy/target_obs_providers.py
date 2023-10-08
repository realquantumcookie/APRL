from rail_walker_interface import JoystickPolicyTargetObservable, BaseWalker, JoystickPolicyTerminationConditionProvider
import gym
import gym.spaces
import numpy as np
from typing import Any, Optional

class JoystickPolicyTargetDeltaYawObservable(JoystickPolicyTargetObservable[BaseWalker]):
    def __init__(self) -> None:
        super().__init__()
        self.obs : float = 0.0

    def get_observation_spec(self) -> gym.Space:
        return gym.spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32)
    
    def step_target_obs(
        self,
        Robot: BaseWalker, 
        target_goal_world_delta: np.ndarray,
        target_goal_local: np.ndarray,
        target_yaw: float, 
        target_delta_yaw: float, 
        target_velocity: float,
        velocity_to_goal: float, 
        change_in_abs_target_delta_yaw : float, 
        target_custom_data: Optional[Any],
        enable_target_custom_obs : bool,
        info_dict: dict[str,Any],
        randomState: np.random.RandomState
    ) -> None:
        self.obs = target_delta_yaw

    def reset_target_obs(
        self, 
        Robot: BaseWalker, 
        target_goal_world_delta: np.ndarray,
        target_goal_local: np.ndarray,
        target_yaw: float, 
        target_delta_yaw: float, 
        target_velocity: float,
        info_dict: dict[str,Any], 
        target_custom_data: Optional[Any],
        enable_target_custom_obs : bool,
        termination_provider_triggered: JoystickPolicyTerminationConditionProvider,
        randomState: np.random.RandomState
    ) -> None:
        self.obs = target_delta_yaw

    def get_observation(self) -> Any:
        return np.array([self.obs], dtype=np.float32)
    
class JoystickPolicyCosSinTargetDeltaYawObservable(JoystickPolicyTargetObservable[BaseWalker]):
    def __init__(self) -> None:
        super().__init__()
        self.obs = np.zeros(2, dtype=np.float32)

    def get_observation_spec(self) -> gym.Space:
        return gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
    
    def step_target_obs(
        self,
        Robot: BaseWalker, 
        target_goal_world_delta: np.ndarray,
        target_goal_local: np.ndarray,
        target_yaw: float, 
        target_delta_yaw: float, 
        target_velocity: float,
        velocity_to_goal: float, 
        change_in_abs_target_delta_yaw : float, 
        target_custom_data: Optional[Any],
        enable_target_custom_obs : bool,
        info_dict: dict[str,Any],
        randomState: np.random.RandomState
    ) -> None:
        self.obs = np.array([np.cos(target_delta_yaw), np.sin(target_delta_yaw)], dtype=np.float32)

    def reset_target_obs(
        self, 
        Robot: BaseWalker, 
        target_goal_world_delta: np.ndarray,
        target_goal_local: np.ndarray,
        target_yaw: float, 
        target_delta_yaw: float, 
        target_velocity: float,
        info_dict: dict[str,Any], 
        target_custom_data: Optional[Any],
        enable_target_custom_obs : bool,
        termination_provider_triggered: JoystickPolicyTerminationConditionProvider,
        randomState: np.random.RandomState
    ) -> None:
        self.obs = np.array([np.cos(target_delta_yaw), np.sin(target_delta_yaw)], dtype=np.float32)

    def get_observation(self) -> Any:
        return self.obs
    
class JoystickPolicyTargetDeltaYawAndDistanceObservable(JoystickPolicyTargetObservable[BaseWalker]):
    def __init__(self) -> None:
        super().__init__()
        self.obs = np.zeros(2, dtype=np.float32)

    def get_observation_spec(self) -> gym.Space:
        return gym.spaces.Box(low=np.array([-np.pi,-np.inf]), high=np.array([np.pi,np.inf]), shape=(2,), dtype=np.float32)
    
    def step_target_obs(
        self,
        Robot: BaseWalker, 
        target_goal_world_delta: np.ndarray,
        target_goal_local: np.ndarray,
        target_yaw: float, 
        target_delta_yaw: float, 
        target_velocity: float,
        velocity_to_goal: float, 
        change_in_abs_target_delta_yaw : float, 
        target_custom_data: Optional[Any],
        enable_target_custom_obs : bool,
        info_dict: dict[str,Any],
        randomState: np.random.RandomState
    ) -> None:
        self.obs = np.array([target_delta_yaw,np.linalg.norm(target_goal_world_delta)], dtype=np.float32)

    def reset_target_obs(
        self, 
        Robot: BaseWalker, 
        target_goal_world_delta: np.ndarray,
        target_goal_local: np.ndarray,
        target_yaw: float, 
        target_delta_yaw: float, 
        target_velocity: float,
        info_dict: dict[str,Any], 
        target_custom_data: Optional[Any],
        enable_target_custom_obs : bool,
        termination_provider_triggered: JoystickPolicyTerminationConditionProvider,
        randomState: np.random.RandomState
    ) -> None:
        self.obs = np.array([target_delta_yaw,np.linalg.norm(target_goal_world_delta)], dtype=np.float32)


    def get_observation(self) -> Any:
        return self.obs