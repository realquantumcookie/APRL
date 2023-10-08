import numpy as np
from rail_walker_interface import Walker3DVelocityEstimatorLocal, BaseWalker, BaseWalkerWithFootContact
from typing import Optional, List
import time
import numba

@numba.jit(nopython=True)
def analytical_leg_jacobian(leg_angles : np.ndarray, l_hip_sign : int):
    """
  Computes the analytical Jacobian.
  Args:
  ` leg_angles: a list of 3 numbers for current abduction, hip and knee angle.
    l_hip_sign: whether it's a RIGHT (0) or LEFT(1) leg.
  """
    l_up = 0.2
    l_low = 0.2
    l_hip = 0.08505 * (-1)**(l_hip_sign + 1)

    t1, t2, t3 = leg_angles[0], leg_angles[1], leg_angles[2]
    l_eff = np.sqrt(l_up**2 + l_low**2 + 2 * l_up * l_low * np.cos(t3))

    t_eff = t2 + t3 / 2
    st1 = np.sin(t1)
    ct1 = np.cos(t1)
    st_eff = np.sin(t_eff)
    ct_eff = np.cos(t_eff)
    st3 = np.sin(t3)

    J = np.zeros((3, 3))
    J[0, 0] = 0
    J[0, 1] = -l_eff * ct_eff
    J[0, 2] = l_low * l_up * st3 * st_eff / l_eff - l_eff * ct_eff / 2
    J[1, 0] = -l_hip * st1 + l_eff * ct1 * ct_eff
    J[1, 1] = -l_eff * st1 * st_eff
    J[1, 2] = -l_low * l_up * st1 * st3 * ct_eff / l_eff - l_eff * st1 * st_eff / 2
    J[2, 0] = l_hip * ct1 + l_eff * st1 * ct_eff
    J[2, 1] = l_eff * st_eff * ct1
    J[2, 2] = l_low * l_up * st3 * ct1 * ct_eff / l_eff + l_eff * st_eff * ct1 / 2
    return J

class Go1ForwardKinematicsVelocityProvider(
    Walker3DVelocityEstimatorLocal[BaseWalker]
):
    def __init__(self):
        super().__init__()
        self._local_velocity = np.zeros(3)
    
    def _recalculate_local_velocity(self, robot: BaseWalker, joints_qvel : np.ndarray):
        foot_contacts = robot.get_foot_contact()
        joints_qpos = robot.get_joint_qpos()
        joints_qvel = robot.get_joint_qvel()

        foot_velocities : List[float] = []
        for leg_idx in range(4):
            if foot_contacts[leg_idx]:
                leg_jacob = analytical_leg_jacobian(
                    joints_qpos[leg_idx * 3 : leg_idx * 3 + 3], 
                    leg_idx
                )
                leg_vel = leg_jacob @ joints_qvel[leg_idx * 3 : leg_idx * 3 + 3]
                foot_velocities.append(leg_vel)
        
        if len(foot_velocities) == 0:
            return
        
        foot_velocities = np.vstack(foot_velocities)
        observed_v = np.mean(foot_velocities, axis=0)
        self._local_velocity = -observed_v
        

    def _reset(self, robot: BaseWalker, frame_quat: Optional[np.ndarray]): 
        self._local_velocity = np.zeros(3)
        self._last_qpos = robot.get_joint_qpos().copy()

    def _step(self, robot: BaseWalker, frame_quat: np.ndarray, dt: float):
        joints_qvel = (robot.get_joint_qpos() - self._last_qpos) / dt
        self._recalculate_local_velocity(robot, joints_qvel)
        self._last_qpos = robot.get_joint_qpos().copy()
    
    def get_3d_local_velocity(self) -> np.ndarray:
        return self._local_velocity