import os

import numpy as np
from .sim_robot import RailSimWalkerDMControl, SIM_ASSET_DIR
from functools import cached_property

_Go1_XML_PATH = os.path.join(SIM_ASSET_DIR, 'robot_assets', 'go1', 'xml', 'go1.xml')

"""
Go1 simulation robot using DM_Control suite
Joint limits are based on https://github.com/unitreerobotics/unitree_legged_sdk/blob/go1/include/unitree_legged_sdk/go1_const.h
They are also cross-checked with XML definition file for go1 (Credit to Kevin for exporting the XML file from Unitree_ROS)
The XML file in unitree_mujoco is actually wrong, it is a copy of a1.xml
"""
class Go1SimWalker(RailSimWalkerDMControl):
    # _INIT_QPOS = np.asarray([0.0, 0.9, -1.8] * 4)
    # _QPOS_OFFSET = np.asarray([0.2, 0.4, 0.4] * 4)

    def __init__(self, Kp=60, Kd=6, action_interpolation : bool = True, limit_action_range : float = 1.0, *args, **kwargs):
        RailSimWalkerDMControl.__init__(
            self,
            XML_FILE = _Go1_XML_PATH,
            Kp = Kp,
            Kd = Kd,
            action_interpolation = action_interpolation,
            limit_action_range = limit_action_range,
            *args,
            **kwargs
        )
        RailSimWalkerDMControl.control_timestep = 0.05
        RailSimWalkerDMControl.control_subtimestep = 0.002

    @cached_property
    def joint_qpos_init(self) -> np.ndarray:
        # shift_delta = np.array([-0.1, -0.1, 0.1])
        # real_shift = shift_delta.repeat(4).reshape((3,4)).T.flatten()
        # real_shift[3::6] = -real_shift[3::6]
        real_shift = np.array([
            -0.1, -0.2, 0.0,
            0.1, -0.2, 0.0,
            -0.1, 0.0, 0.0,
            0.1, 0.0, 0.0
        ])
        return np.array([0.0 / 180 * np.pi, 45.0/180*np.pi, -90.0 / 180 * np.pi] * 4) + real_shift

    @cached_property
    def joint_qpos_sitting(self) -> np.ndarray:
        return np.array([0.0 / 180 * np.pi, 70.0/180*np.pi, -150.0 / 180 * np.pi] * 4)
    
    @cached_property
    def joint_qpos_offset(self) -> np.ndarray:
        return np.array([0.2, 0.4, 0.4] * 4)

    @cached_property
    def joint_qpos_mins(self) -> np.ndarray:
        return np.asarray([-1.047, -0.663, -2.721] * 4)

    @cached_property
    def joint_qpos_maxs(self) -> np.ndarray:
        return np.asarray([1.047, 2.966, -0.837] * 4)

