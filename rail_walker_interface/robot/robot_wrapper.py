from typing import Optional
from .robot import BaseWalker
from typing import TypeVar, Generic, Any
import numpy as np
import gym.spaces

_WalkerCls = TypeVar("_WalkerCls", bound=BaseWalker)
class WalkerWrapper(BaseWalker, Generic[_WalkerCls]):
    def __init__(self, robot : BaseWalker):
        self.robot = robot
    
    def __getattr__(self, name : str):
        if "robot" in self.__dict__ and hasattr(self.robot, name):
            return getattr(self.robot, name)
        elif name in self.__dict__:
            return self.__dict__[name]
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def __setattr__(self, name : str, value : Any):
        if hasattr(self, 'robot') and hasattr(self.robot, name):
            setattr(self.robot, name, value)
        else:
            self.__dict__[name] = value

    @property
    def is_real_robot(self) -> bool:
        return self.robot.is_real_robot

    @property
    def power_protect_factor(self) -> float:
        return self.robot.power_protect_factor
    
    @power_protect_factor.setter
    def power_protect_factor(self, value: float) -> None:
        self.robot.power_protect_factor = value

    @property
    def control_timestep(self) -> float:
        return self.robot.control_timestep
    
    @property
    def action_interpolation(self) -> bool:
        return self.robot.action_interpolation

    @property
    def control_subtimestep(self) -> float:
        return self.robot.control_subtimestep

    def receive_observation(self) -> bool:
        return self.robot.receive_observation()

    @property
    def joint_qpos_init(self) -> np.ndarray:
        return self.robot.joint_qpos_init

    @property
    def joint_qpos_sitting(self) -> np.ndarray:
        return self.robot.joint_qpos_sitting

    @property
    def joint_qpos_crouch(self) -> np.ndarray:
        return self.robot.joint_qpos_crouch

    @property
    def joint_qpos_offset(self) -> np.ndarray:
        return self.robot.joint_qpos_offset

    @property
    def joint_qpos_mins(self) -> np.ndarray:
        return self.robot.joint_qpos_mins

    @property
    def joint_qpos_maxs(self) -> np.ndarray:
        return self.robot.joint_qpos_maxs

    def reset(self) -> None:
        self.robot.reset()

    def get_3d_linear_velocity(self) -> np.ndarray:
        return self.robot.get_3d_linear_velocity()

    def get_3d_local_velocity(self) -> np.ndarray:
        return self.robot.get_3d_local_velocity()

    def get_3d_angular_velocity(self) -> np.ndarray:
        return self.robot.get_3d_angular_velocity()

    def get_framequat_wijk(self) -> np.ndarray:
        return self.robot.get_framequat_wijk()

    def get_roll_pitch_yaw(self) -> np.ndarray:
        return self.robot.get_roll_pitch_yaw()

    def get_last_observation(self) -> Optional[Any]:
        return self.robot.get_last_observation()

    def get_3d_acceleration_local(self) -> np.ndarray:
        return self.robot.get_3d_acceleration_local()

    def get_joint_qpos(self) -> np.ndarray:
        return self.robot.get_joint_qpos()

    def get_joint_qvel(self) -> np.ndarray:
        return self.robot.get_joint_qvel()
    
    def get_joint_qacc(self) -> np.ndarray:
        return self.robot.get_joint_qacc()

    def get_joint_torques(self) -> np.ndarray:
        return self.robot.get_joint_torques()

    def _apply_action(self, action: np.ndarray) -> bool:
        return self.robot._apply_action(action)

    def close(self) -> None:
        self.robot.close()

    @property
    def action_qpos_mins(self) -> np.ndarray:
        return self.robot.action_qpos_mins
    
    @property
    def action_qpos_maxs(self) -> np.ndarray:
        return self.robot.action_qpos_maxs

    def apply_action(self, action: np.ndarray) -> bool:
        return self.robot.apply_action(action)

    def can_apply_action(self) -> bool:
        return self.robot.can_apply_action()

    def async_apply_action(self, action: np.ndarray) -> bool:
        return self.robot.async_apply_action(action)

    @property
    def joint_nums(self) -> int:
        return self.robot.joint_nums
    
    @property
    def action_spec(self) -> gym.spaces.Box:
        return self.robot.action_spec

    def unwrapped(self):
        return self.robot.unwrapped()