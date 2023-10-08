import pyrealsense2 as rs
from rail_walker_interface import BaseWalker, Walker3DVelocityEstimator
from typing import Optional
import multiprocessing
import numpy as np
import transforms3d as tr3d
import time

class DummyVelocityEstimator(Walker3DVelocityEstimator):
    def __init__(
        self
    ):
        super().__init__()
        self.velocity = np.zeros(3)

    def reset(self, robot: BaseWalker, frame_quat: Optional[np.ndarray]):
        self.velocity[:] = 0.0

    def step(self, robot: BaseWalker, frame_quat: Optional[np.ndarray], dt: float):
        self.velocity[:] = 0.0

    def get_3d_linear_velocity(self) -> np.ndarray:
        return self.velocity

    def close(self) -> None:
        pass