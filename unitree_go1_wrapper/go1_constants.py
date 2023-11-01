import numpy as np

"""
From go1_const.h
"""

GO1_HIP_MAX   = 1.047    # unit:radian ( = 60   degree)
GO1_HIP_MIN   = -1.047   # unit:radian ( = -60  degree)
GO1_THIGH_MAX = 2.966    # unit:radian ( = 170  degree)
GO1_THIGH_MIN = -0.663   # unit:radian ( = -38  degree)
GO1_CALF_MAX  = -0.837   # unit:radian ( = -48  degree)
GO1_CALF_MIN  = -2.721   # unit:radian ( = -156 degree)
GO1_HIP_INIT = 0.0 / 180 * np.pi
GO1_THIGH_INIT = 45.0 / 180 * np.pi
GO1_CALF_INIT = -90.0 / 180 * np.pi