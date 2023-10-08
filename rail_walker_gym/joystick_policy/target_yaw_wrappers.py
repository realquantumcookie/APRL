from typing import Any, Optional
from dm_control.mujoco.engine import Physics as EnginePhysics
import gym
from rail_mujoco_walker.tasks.joystick_task import JoystickPolicyDMControlTask
from rail_walker_interface import JoystickPolicy, JoystickPolicyTargetProvider, BaseWalker, JoystickPolicyTerminationConditionProvider
import numpy as np
from dm_control.mjcf.physics import Physics
import transforms3d as tr3d
import typing
from rail_mujoco_walker import add_arrow_to_mjv_scene, JoystickPolicyProviderWithDMControl
from mujoco import MjvScene

def normalize_rad(rad : float) -> float:
    return (rad + np.pi) % (2 * np.pi) - np.pi

_RobotType = typing.TypeVar("_RobotType", bound=BaseWalker)
class JoystickPolicyTargetProviderWrapper(typing.Generic[_RobotType], JoystickPolicyTargetProvider[_RobotType], JoystickPolicyProviderWithDMControl):
    def __init__(
        self,
        provider: JoystickPolicyTargetProvider[_RobotType]
    ):
        JoystickPolicyTargetProvider.__init__(self)
        JoystickPolicyProviderWithDMControl.__init__(self)
        self.provider = provider
        self.original_target : np.ndarray = np.zeros(2)
        self.target : np.ndarray = np.zeros(2)
    
    def get_target_goal_world_delta(self, Robot: _RobotType) -> np.ndarray:
        return self.target
    
    def get_target_velocity(self, Robot: _RobotType) -> float:
        return self.provider.get_target_velocity(Robot)
    
    def is_target_velocity_fixed(self) -> bool:
        return self.provider.is_target_velocity_fixed()

    def get_target_custom_data(self) -> Any | None:
        return self.provider.get_target_custom_data()

    def get_target_custom_data_observable(self) -> Any | None:
        return self.provider.get_target_custom_data_observable()

    def get_target_custom_data_observable_spec(self) -> gym.Space | None:
        return self.provider.get_target_custom_data_observable_spec()

    def calculate_new_target(
        self,
        Robot: _RobotType,
        info_dict: dict[str,typing.Any],
        randomState: np.random.RandomState
    ):
        return self.original_target

    def step(
        self, 
        Robot: _RobotType, 
        info_dict: dict[str,typing.Any], 
        randomState : np.random.RandomState
    ) -> None:
        self.provider.step(
            Robot,
            info_dict,
            randomState
        )
        self.original_target = self.provider.get_target_goal_world_delta(Robot)
        self.target = self.calculate_new_target(Robot, info_dict, randomState)

    def reset(
        self, 
        Robot: _RobotType, 
        info_dict: dict[str,typing.Any], 
        termination_provider_triggered: JoystickPolicyTerminationConditionProvider,
        randomState: np.random.RandomState
    ) -> None:
        self.provider.reset(
            Robot,
            info_dict,
            termination_provider_triggered,
            randomState
        )
        self.original_target = self.provider.get_target_goal_world_delta(Robot)
        self.target = self.calculate_new_target(Robot, info_dict, randomState)

    def render_scene_callback(self, task: JoystickPolicyDMControlTask, physics: Physics, scene: MjvScene) -> None:
        if hasattr(self.provider, "render_scene_callback"):
            self.provider.render_scene_callback(task, physics, scene)
        
        orig_target_unit_3d = np.array([*self.original_target, 0.0])
        orig_target_unit_3d /= np.linalg.norm(orig_target_unit_3d)

        if not np.isclose(self.target, self.original_target).all():
            root_arrow = task._arrow_root + np.array([0.0,0.0,0.25])
            add_arrow_to_mjv_scene(
                scene,
                root_arrow,
                root_arrow + orig_target_unit_3d * 0.5,
                0.01,
                np.array([0.0, 1.0, 1.0, 1.0]) #cyan for original target
            )


_LimitedRobotType = typing.TypeVar("_LimitedRobotType", bound=BaseWalker)
class JoystickPolicyLimitedTargetProvider(typing.Generic[_LimitedRobotType], JoystickPolicyTargetProviderWrapper[_LimitedRobotType]):
    def __init__(
        self, 
        provider: JoystickPolicyTargetProvider[_LimitedRobotType],
        max_delta_angle: float = 30.0 / 180 * np.pi, # 30.0 degrees max
    ):
        JoystickPolicyTargetProviderWrapper.__init__(self, provider)
        self.max_delta_angle = max_delta_angle
    
    def calculate_new_target(
        self,
        Robot: _RobotType,
        info_dict: dict[str,typing.Any],
        randomState: np.random.RandomState
    ):
        roll, pitch, yaw = Robot.get_roll_pitch_yaw()
        original_target_yaw = np.arctan2(self.original_target[1], self.original_target[0])
        target_delta_yaw = normalize_rad(original_target_yaw - yaw)

        clipped_target_delta_yaw = np.clip(target_delta_yaw, -self.max_delta_angle, self.max_delta_angle)
        
        target = np.array([
            np.cos(yaw + clipped_target_delta_yaw),
            np.sin(yaw + clipped_target_delta_yaw)
        ]) # unit vector pointing in the clipped direction
        target *= np.maximum(np.inner(self.original_target, target), 0.1) # project the original target onto the clipped direction
        return target

_SmoothedRobotType = typing.TypeVar("_SmoothedRobotType", bound=BaseWalker)
class JoystickPolicySmoothedTargetProvider(typing.Generic[_SmoothedRobotType], JoystickPolicyTargetProviderWrapper[_SmoothedRobotType]):
    def __init__(
        self,
        provider: JoystickPolicyTargetProvider,
        max_rate: float = 30.0 / 180 * np.pi, # 30.0 degrees max per step
    ):
        JoystickPolicyTargetProviderWrapper.__init__(self, provider)
        self.max_rate = max_rate

    def calculate_new_target(
        self,
        Robot: _RobotType,
        info_dict: dict[str,typing.Any],
        randomState: np.random.RandomState
    ):
        original_target_yaw = np.arctan2(self.original_target[1], self.original_target[0])
        last_target_yaw = np.arctan2(self.target[1], self.target[0])
        change_in_target_yaw = normalize_rad(original_target_yaw - last_target_yaw)

        clipped_change_in_target_yaw = np.clip(change_in_target_yaw, -self.max_rate, self.max_rate)
        new_target_yaw = normalize_rad(last_target_yaw + clipped_change_in_target_yaw)
        
        target = np.array([
            np.cos(new_target_yaw),
            np.sin(new_target_yaw)
        ]) # unit vector pointing in the clipped direction
        target *= np.maximum(np.inner(self.original_target, target), 0.1) # project the original target onto the clipped direction
        return target