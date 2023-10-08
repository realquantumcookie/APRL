import gym
import numpy as np
from typing import Optional, Any, TypeVar, Generic, Callable
from ..robot.robot import BaseWalker

_RobotClsTerminationT = TypeVar("_RobotClsTerminationT", bound=BaseWalker)
class JoystickPolicyTerminationConditionProvider(Generic[_RobotClsTerminationT]):
    def should_terminate(self) -> bool:
        raise NotImplementedError()
    
    def step_termination_condition(
            self, 
            Robot: _RobotClsTerminationT, 
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
        pass

    def reset_termination_condition(
        self, 
        Robot: _RobotClsTerminationT,
        info_dict: dict[str,Any], 
        termination_provider_triggered,
        randomState: np.random.RandomState
    ) -> None:
        pass

_RobotClsTargetObsT = TypeVar("_RobotClsTargetObsT", bound=BaseWalker)
class JoystickPolicyTargetObservable(Generic[_RobotClsTargetObsT]):
    def get_observation_spec(self) -> gym.Space:
        raise NotImplementedError()
    
    def step_target_obs(
        self,
        Robot: _RobotClsTargetObsT, 
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
        pass

    def reset_target_obs(
        self, 
        Robot: _RobotClsTargetObsT, 
        target_goal_world_delta: np.ndarray,
        target_goal_local: np.ndarray,
        target_yaw : float,
        target_delta_yaw: float, 
        target_velocity: float,
        info_dict: dict[str,Any], 
        target_custom_data: Optional[Any],
        enable_target_custom_obs : bool,
        termination_provider_triggered: JoystickPolicyTerminationConditionProvider,
        randomState: np.random.RandomState
    ) -> None:
        pass

    def get_observation(self) -> Any:
        raise NotImplementedError()

_RobotClsTargetProviderT = TypeVar("_RobotClsTargetProviderT", bound=BaseWalker)
class JoystickPolicyTargetProvider(Generic[_RobotClsTargetProviderT]):
    def get_target_goal_world_delta(self, Robot: _RobotClsTargetProviderT) -> np.ndarray:
        raise NotImplementedError()
    
    def get_target_velocity(self, Robot: _RobotClsTargetProviderT) -> float:
        return 0.5
    
    def is_target_velocity_fixed(self) -> bool:
        return True

    def get_target_custom_data(self) -> Optional[Any]:
        return None
    
    def get_target_custom_data_observable_spec(self) -> Optional[gym.Space]:
        return None
    
    def get_target_custom_data_observable(self) -> Optional[Any]:
        return None
    
    def has_target_changed(self) -> bool:
        return False

    def step_target(
        self, 
        Robot: _RobotClsTargetProviderT, 
        info_dict: dict[str,Any], 
        randomState : np.random.RandomState
    ) -> None:
        pass

    def after_step_target(
        self, 
        Robot: _RobotClsTargetProviderT, 
        info_dict: dict[str,Any], 
        randomState : np.random.RandomState
    ) -> None:
        pass

    def reset_target(
        self, 
        Robot: _RobotClsTargetProviderT, 
        info_dict: dict[str,Any], 
        termination_provider_triggered: JoystickPolicyTerminationConditionProvider,
        randomState: np.random.RandomState
    ) -> None:
        pass

_RobotClsRewardProviderT = TypeVar("_RobotClsRewardProviderT", bound=BaseWalker)
class JoystickPolicyRewardProvider(Generic[_RobotClsRewardProviderT]):
    def get_reward(self) -> float:
        raise NotImplementedError()

    def step_reward(
        self, 
        Robot: _RobotClsRewardProviderT, 
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
        pass

    def reset_reward(
        self, 
        Robot: _RobotClsRewardProviderT, 
        info_dict: dict[str,Any], 
        termination_provider_triggered: JoystickPolicyTerminationConditionProvider,
        randomState: np.random.RandomState
    ) -> None:
        pass

_RobotClsResetterT = TypeVar("_RobotClsResetterT", bound=BaseWalker)
class JoystickPolicyResetter(Generic[_RobotClsResetterT]):
    def perform_reset(
        self, 
        Robot: _RobotClsResetterT, 
        info_dict: dict[str,Any], 
        termination_provider_triggered: JoystickPolicyTerminationConditionProvider,
        randomState: np.random.RandomState
    ) -> None:
        pass

    def step_resetter(
        self, 
        Robot: _RobotClsTerminationT, 
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
        pass