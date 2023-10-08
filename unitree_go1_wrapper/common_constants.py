from enum import Enum
import math

"""
From quadruped.h
"""
class LegIndex(Enum):
    LEG_INDEX_FR = 0
    LEG_INDEX_FL = 1
    LEG_INDEX_RR = 2
    LEG_INDEX_RL = 3

class JointIndex(Enum):
    FR_0 = 0
    FR_1 = 1
    FR_2 = 2

    FL_0 = 3
    FL_1 = 4
    FL_2 = 5

    RR_0 = 6
    RR_1 = 7
    RR_2 = 8

    RL_0 = 9
    RL_1 = 10
    RL_2 = 11

"""
From comm.h
"""
class ControlLevel(Enum):
    HIGH = 0xee
    LOW = 0xff
    TRIGGER = 0xf0

"""
might be something else, see https://github.com/unitreerobotics/unitree_legged_sdk/blob/v3.8.0/example_py/example_position.py
also see https://github.com/unitreerobotics/unitree_legged_sdk/blob/v3.8.0/include/unitree_legged_sdk/comm.h
"""
POS_STOP_F = math.pow(10,9) #2.146e9 

VEL_STOP_F = 16000.0