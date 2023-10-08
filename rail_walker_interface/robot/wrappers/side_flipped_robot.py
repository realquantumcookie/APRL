import numpy as np
from ..robot_wrapper import WalkerWrapper
from functools import cached_property
import transforms3d as tr3d

class SideFlippedWalkerWrapper(WalkerWrapper):
    def receive_observation(self) -> bool:
        ret = self.robot.receive_observation()
        self._local_velocity = self.robot.get_3d_local_velocity()
        self._local_velocity[1] *= -1
        self._linear_velocity = tr3d.quaternions.rotate_vector(self._local_velocity, self.robot.get_framequat_wijk())
        self._local_accel = self.robot.get_3d_acceleration_local()
        self._local_accel[1] *= -1
        self._angular_velocity = self.robot.get_3d_angular_velocity()
        self._angular_velocity[2] *= -1
        return ret

    def get_joint_qpos(self) -> np.ndarray:
        return self._transform_joint_readings(self.robot.get_joint_qpos())

    def _transform_joint_readings(self, readings : np.ndarray) -> np.ndarray:
        return_readings = np.zeros_like(readings)
        return_readings[self.joint_indices_right_hips] = -readings[self.joint_indices_left_hips]
        return_readings[self.joint_indices_left_hips] = -readings[self.joint_indices_right_hips]
        return_readings[self.joint_indices_right_thighs] = readings[self.joint_indices_left_thighs]
        return_readings[self.joint_indices_left_thighs] = readings[self.joint_indices_right_thighs]
        return_readings[self.joint_indices_right_calfs] = readings[self.joint_indices_left_calfs]
        return_readings[self.joint_indices_left_calfs] = readings[self.joint_indices_right_calfs]
        return return_readings

    def get_joint_qvel(self) -> np.ndarray:
        return self._transform_joint_readings(self.robot.get_joint_qvel())
    
    def get_joint_qacc(self) -> np.ndarray:
        return self._transform_joint_readings(self.robot.get_joint_qacc())
    
    def get_joint_torques(self) -> np.ndarray:
        return self._transform_joint_readings(self.robot.get_joint_torques())
    
    @cached_property
    def joint_qpos_init(self) -> np.ndarray:
        return self._transform_joint_readings(self.robot.joint_qpos_init)
    
    @cached_property
    def joint_qpos_sitting(self) -> np.ndarray:
        return self._transform_joint_readings(self.robot.joint_qpos_sitting)
    
    @cached_property
    def joint_qpos_offset(self) -> np.ndarray:
        return self._transform_joint_readings(self.robot.joint_qpos_offset)
    
    @cached_property
    def joint_qpos_crouch(self) -> np.ndarray:
        return self._transform_joint_readings(self.robot.joint_qpos_crouch)
    
    @cached_property
    def joint_qpos_mins(self) -> np.ndarray:
        return self._transform_joint_readings(self.robot.joint_qpos_mins)
    
    @cached_property
    def joint_qpos_maxs(self) -> np.ndarray:
        return self._transform_joint_readings(self.robot.joint_qpos_maxs)
    
    @property
    def action_qpos_mins(self) -> np.ndarray:
        return self._transform_joint_readings(self.robot.action_qpos_mins)
    
    @property
    def action_qpos_maxs(self) -> np.ndarray:
        return self._transform_joint_readings(self.robot.action_qpos_maxs)
    
    def get_3d_linear_velocity(self) -> np.ndarray:
        return self._linear_velocity

    def get_3d_local_velocity(self) -> np.ndarray:
        return self._local_velocity
    
    def get_3d_acceleration_local(self) -> np.ndarray:
        return self._local_accel

    def get_3d_angular_velocity(self) -> np.ndarray:
        return self._angular_velocity
    
    @cached_property
    def joint_indices_left_hips(self):
        return np.array([3, 9])
    
    @cached_property
    def joint_indices_left_thighs(self):
        return np.array([4, 10])
    
    @cached_property
    def joint_indices_left_calfs(self):
        return np.array([5, 11])
    
    @cached_property
    def joint_indices_right_hips(self):
        return np.array([0, 6])
    
    @cached_property
    def joint_indices_right_thighs(self):
        return np.array([1, 7])
    
    @cached_property
    def joint_indices_right_calfs(self):
        return np.array([2, 8])
    
    