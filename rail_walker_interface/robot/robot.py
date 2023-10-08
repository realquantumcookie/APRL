from typing import Generic, Optional, TypeVar
import numpy as np
from functools import cached_property
import gym
import gym.spaces
import transforms3d as tr
import time


_ObsT = TypeVar("_ObsT")
class BaseWalker(Generic[_ObsT]):
    def __init__(
        self, 
        name: Optional[str] = "robot", 
        Kp: float = 5,
        Kd: float = 1,
        force_real_control_timestep : bool = False,
        limit_action_range : float = 1.0,
        power_protect_factor : float = 0.1
    ):
        assert limit_action_range > 0 and limit_action_range <= 1.0
        self.name = name
        self.Kp = Kp
        self.Kd = Kd
        self.force_real_control_timestep = force_real_control_timestep
        self._last_control_t = 0.0
        self.limit_action_range = limit_action_range
        self._power_protect_factor = power_protect_factor

    @property
    def is_real_robot(self) -> bool:
        return False

    @property
    def power_protect_factor(self) -> float:
        return self._power_protect_factor
    
    @power_protect_factor.setter
    def power_protect_factor(self, value: float) -> None:
        assert value >= 0 and value <= 1.0
        self._power_protect_factor = value

    """
    The control_timestep is the time interval between two consecutive model control actions.
    """
    @property
    def control_timestep(self) -> float:
        pass
    
    @property
    def action_interpolation(self) -> bool:
        pass

    """
    The control_subtimestep is the time interval between two consecutive internal control actions. It will also be the physics timestep if in simulation.
    """
    @property
    def control_subtimestep(self) -> float:
        pass

    def receive_observation(self) -> bool:
        pass

    @property
    def joint_qpos_init(self) -> np.ndarray:
        pass

    @property
    def joint_qpos_sitting(self) -> np.ndarray:
        pass

    @cached_property
    def joint_qpos_crouch(self) -> np.ndarray:
        return (self.joint_qpos_init + self.joint_qpos_sitting) / 2.0

    """
    This property will be used to determine the standing range of qpos of the robot.
    """
    @property
    def joint_qpos_offset(self) -> np.ndarray:
        pass

    @property
    def joint_qpos_mins(self) -> np.ndarray:
        pass

    @property
    def joint_qpos_maxs(self) -> np.ndarray:
        pass

    def reset(self) -> None:
        pass

    def get_3d_linear_velocity(self) -> np.ndarray:
        pass

    def get_3d_local_velocity(self) -> np.ndarray:
        pass

    def get_3d_angular_velocity(self) -> np.ndarray:
        pass

    def get_framequat_wijk(self) -> np.ndarray:
        pass

    def get_roll_pitch_yaw(self) -> np.ndarray:
        pass

    def get_last_observation(self) -> Optional[_ObsT]:
        pass

    def get_3d_acceleration_local(self) -> np.ndarray:
        pass

    def get_joint_qpos(self) -> np.ndarray:
        pass

    def get_joint_qvel(self) -> np.ndarray:
        pass

    def get_joint_qacc(self) -> np.ndarray:
        pass

    def get_joint_torques(self) -> np.ndarray:
        pass

    def _apply_action(self, action: np.ndarray) -> bool:
        pass

    def close(self) -> None:
        pass

    def __del__(self):
        self.close()
    
    @property
    def action_qpos_mins(self) -> np.ndarray:
        # delta = -np.minimum(np.abs(self.joint_qpos_mins - self.joint_qpos_init), np.abs(self.joint_qpos_maxs - self.joint_qpos_init))
        # return delta * self.limit_action_range + self.joint_qpos_init
        return (self.joint_qpos_mins - self.joint_qpos_init) * self.limit_action_range + self.joint_qpos_init
    
    @property
    def action_qpos_maxs(self) -> np.ndarray:
        # delta = np.minimum(np.abs(self.joint_qpos_mins - self.joint_qpos_init), np.abs(self.joint_qpos_maxs - self.joint_qpos_init))
        # return delta * self.limit_action_range + self.joint_qpos_init
        return (self.joint_qpos_maxs - self.joint_qpos_init) * self.limit_action_range + self.joint_qpos_init

    def apply_action(self, action: np.ndarray) -> bool:
        action = np.clip(action, self.action_qpos_mins, self.action_qpos_maxs)
        
        if not self.force_real_control_timestep:
            return self._apply_action(action)
        else:
            t = time.time()
            dt = t - self._last_control_t
            if dt >= self.control_timestep:
                self._last_control_t = t
                return self._apply_action(action)
            else:
                time_to_sleep = self.control_timestep - dt
                time.sleep(time_to_sleep)
                self._last_control_t = t + time_to_sleep
                return self._apply_action(action)

    def can_apply_action(self) -> bool:
        t = time.time()
        dt = t - self._last_control_t
        if (not self.force_real_control_timestep) or dt >= self.control_timestep:
            return True
        else:
            return False

    def async_apply_action(self, action: np.ndarray) -> bool:
        if self.can_apply_action():
            self._last_control_t = time.time()
            return self._apply_action(action)
        else:
            return False

    @cached_property
    def joint_nums(self) -> int:
        return len(self.joint_qpos_init)
    
    @cached_property
    def action_spec(self) -> gym.spaces.Box:
        return gym.spaces.Box(
            low=self.joint_qpos_mins, 
            high=self.joint_qpos_maxs, 
            shape=(self.joint_nums,),
            dtype=np.float32
        )

    def unwrapped(self):
        return self

class BaseWalkerWithFootContact:
    def get_foot_contact(self) -> np.ndarray:
        pass

    def get_foot_force(self) -> np.ndarray:
        pass

    def get_foot_force_norm(self) -> np.ndarray:
        pass

class BaseWalkerWithJointTemperatureSensor():
    def get_joint_temperature_celsius(self) -> np.ndarray:
        pass

class BaseWalkerWithJoystick:
    def get_joystick_values(self) -> list[np.ndarray]:
        pass
 
class BaseWalkerLocalizable():
    def get_3d_location(self) -> np.ndarray:
        pass

class BaseWalkerInSim(BaseWalkerLocalizable):
    def set_3d_location(self, target_location : np.ndarray) -> np.ndarray:
        pass

    def reset_2d_location(self, target_location : np.ndarray, target_quaternion : Optional[np.ndarray] = None, target_qpos : Optional[np.ndarray] = None) -> np.ndarray:
        pass

    def set_framequat_wijk(self, framequat_wijk: np.ndarray) -> None:
        pass

    def set_roll_pitch_yaw(self, roll: float, pitch: float, yaw: float) -> None:
        pass

_RobotCls = TypeVar("_RobotCls", bound=BaseWalker)
class Walker3DVelocityEstimator(Generic[_RobotCls]):
    def __init__(self):
        self._enabled = True

    def reset(self, robot: _RobotCls, frame_quat: Optional[np.ndarray]):
        pass

    def step(self, robot: _RobotCls, frame_quat: Optional[np.ndarray], dt: float):
        pass

    def get_3d_linear_velocity(self) -> np.ndarray:
        pass

    def close(self) -> None:
        pass

    @property
    def enabled(self) -> bool:
        return self._enabled
    
    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value
    
    def __del__(self):
        self.close()

_RobotClsFramequatEst = TypeVar("_RobotClsFramequatEst", bound=BaseWalker)
class Walker3DFrameQuatEstimator(Generic[_RobotClsFramequatEst]):
    def reset(self, robot: _RobotClsFramequatEst, frame_quat: Optional[np.ndarray]):
        pass

    def step(self, robot: _RobotClsFramequatEst, frame_quat: Optional[np.ndarray], dt: float):
        pass

    def get_framequat_wijk(self) -> np.ndarray:
        pass
    
    def close(self) -> None:
        pass

    @property
    def enabled(self) -> bool:
        return True

    def __del__(self):
        self.close()
    

_RobotClsLocalEst = TypeVar("_RobotClsLocalEst", bound=BaseWalker)
class Walker3DVelocityEstimatorLocal(Walker3DVelocityEstimator[_RobotClsLocalEst], Generic[_RobotClsLocalEst]):
    def __init__(self):
        self._global_velocity = np.zeros(3)

    def reset(self, robot: _RobotClsLocalEst, frame_quat: Optional[np.ndarray]):
        self._global_velocity = np.zeros(3)
        self._reset(robot, frame_quat)

    def step(self, robot: _RobotClsLocalEst, frame_quat: Optional[np.ndarray], dt: float):
        self._step(robot, frame_quat, dt)
        local_velocity = self.get_3d_local_velocity()
        self._global_velocity = tr.quaternions.rotate_vector(local_velocity, frame_quat)

    def get_3d_linear_velocity(self) -> np.ndarray:
        return self._global_velocity
    
    def _reset(self, robot: _RobotClsLocalEst, frame_quat: Optional[np.ndarray]): 
        pass

    def _step(self, robot: _RobotClsLocalEst, frame_quat: np.ndarray, dt: float):
        raise NotImplementedError()
    
    def get_3d_local_velocity(self) -> np.ndarray:
        raise NotImplementedError()

_RobotClsPositionEst = TypeVar("_RobotClsPositionEst", bound=BaseWalker)
class Walker3DLocationEstimator(Generic[_RobotClsPositionEst], Walker3DVelocityEstimator[_RobotClsPositionEst]):
    def __init__(self) -> None:
        Walker3DVelocityEstimator.__init__(self)
        self._last_location = None
        self._last_velocity = None

    def reset(self, robot: _RobotCls, frame_quat: Optional[np.ndarray]):
        self._last_location = self._estimate_location(robot,frame_quat)
        self._last_velocity = np.zeros(3)

    def step(self, robot: _RobotCls, frame_quat: Optional[np.ndarray], dt: float):
        location = self._estimate_location(robot, frame_quat)
        self._last_velocity = (location - self._last_location) / dt
        self._last_location = location

    def get_3d_linear_velocity(self) -> np.ndarray:
        return self._last_velocity

    def get_3d_location(self) -> np.ndarray:
        return self._last_location
    
    def _estimate_location(self, robot: _RobotCls, frame_quat : Optional[np.ndarray]) -> np.ndarray:
        raise NotImplementedError()

    def close(self) -> None:
        pass

    def __del__(self):
        self.close()
    