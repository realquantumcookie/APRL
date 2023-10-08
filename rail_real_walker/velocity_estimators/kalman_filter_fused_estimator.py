from typing import Optional
from .foot_contact_estimator import Go1ForwardKinematicsVelocityProvider
from rail_walker_interface import Walker3DVelocityEstimatorLocal, BaseWalker
from filterpy.kalman import KalmanFilter
import numpy as np
from typing import Optional
import transforms3d as tr3d

DEFAULT_ACCELEROMETER_NOISE = 0.0306
DEFAULT_FK_NOISE = 0.00621
GRAVITY_VECTOR = np.array([0, 0, 9.81])

class KalmanFilterFusedVelocityEstimator(Walker3DVelocityEstimatorLocal):
    def __init__(
        self, 
        accelerometer_variance: float = DEFAULT_ACCELEROMETER_NOISE,
        sensor_variance: float = DEFAULT_FK_NOISE,
        sensor : Walker3DVelocityEstimatorLocal = Go1ForwardKinematicsVelocityProvider(),
        initial_variance: float = 0.1
    ):
        super().__init__()
        self.sensor = sensor

        self._initial_variance = initial_variance
        self.filter = KalmanFilter(dim_x=3, dim_z=3, dim_u=3)
        
        self.filter.x = np.zeros(3)
        self.filter.P = np.eye(3) * self._initial_variance
        self.filter.Q = np.eye(3) * accelerometer_variance
        self.filter.R = np.eye(3) * sensor_variance

        self.filter.H = np.eye(3) # measurement function, y = Hx
        self.filter.F = np.eye(3) # state transition matrix
        self.filter.B = np.eye(3)

        self._estimated_local_velocity = np.zeros(3)
    
    def _update_velocity_estimate(self, robot: BaseWalker, dt: float):

        raw_acceleration = robot.get_3d_acceleration_local()
        gravity_local = tr3d.quaternions.rotate_vector(
            GRAVITY_VECTOR,
            tr3d.quaternions.qinverse(robot.get_framequat_wijk())
        )
        calibrated_local_acc = raw_acceleration - gravity_local
        self.filter.predict(u=calibrated_local_acc * dt)
        self.filter.update(self.sensor.get_3d_local_velocity())
        self._estimated_local_velocity = self.filter.x.copy()

    def _reset(self, robot: BaseWalker, frame_quat: Optional[np.ndarray]):
        self.sensor._reset(robot, frame_quat)
        
        self.filter.x = np.zeros(3)
        # self.filter.P = np.eye(3) * self._initial_variance

        self._estimated_local_velocity = np.zeros(3)

    def _step(self, robot: BaseWalker, frame_quat: np.ndarray, dt: float):
        self.sensor._step(robot, frame_quat, dt)
        self._update_velocity_estimate(robot, dt)
    
    def get_3d_local_velocity(self) -> np.ndarray:
        return self._estimated_local_velocity