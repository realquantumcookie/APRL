import os

import numpy as np
from .sim_robot import RailSimWalkerDMControl, SIM_ASSET_DIR
from functools import cached_property

_A1_XML_PATH = os.path.join(SIM_ASSET_DIR, 'robot_assets', 'a1', 'xml', 'a1.xml')

"""
A1 simulation robot using DM_Control suite
Joint limits are based on https://github.com/unitreerobotics/unitree_legged_sdk/blob/v3.3.1/include/unitree_legged_sdk/a1_const.h
They are also cross-checked with XML definition file for a1
"""
class A1SimWalker(RailSimWalkerDMControl):
    # _INIT_QPOS = np.asarray([0.0, 0.9, -1.8] * 4)
    # _QPOS_OFFSET = np.asarray([0.2, 0.4, 0.4] * 4)
    # _INIT_QPOS = np.asarray([-0.1, 0.5, -1.4] * 4)

    def __init__(self, Kp=60, Kd=6, action_interpolation : bool = True, limit_action_range : float = 1.0, *args, **kwargs):
        RailSimWalkerDMControl.__init__(
            self,
            XML_FILE = _A1_XML_PATH,
            Kp = Kp,
            Kd = Kd,
            action_interpolation = action_interpolation,
            limit_action_range=limit_action_range,
            *args,
            **kwargs
        )

    @cached_property
    def joint_qpos_init(self) -> np.ndarray:
        return np.asarray([0.0 / 180 * np.pi, 45.0/180*np.pi, -90.0 / 180 * np.pi] * 4)
    
    @cached_property
    def joint_qpos_sitting(self) -> np.ndarray:
        return np.array([0.0 / 180 * np.pi, 70.0/180*np.pi, -150.0 / 180 * np.pi] * 4)

    @cached_property
    def joint_qpos_offset(self) -> np.ndarray:
        return np.array([0.2, 0.4, 0.4] * 4)

    @cached_property
    def joint_qpos_mins(self) -> np.ndarray:
        return np.asarray([-0.802, -1.05, -2.7] * 4)

    @cached_property
    def joint_qpos_maxs(self) -> np.ndarray:
        return np.asarray([0.802, 4.19, -0.916] * 4)