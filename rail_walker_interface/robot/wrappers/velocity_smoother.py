from rail_walker_interface.robot.robot import BaseWalker
from ..robot_wrapper import WalkerWrapper
from ..robot import BaseWalker
from collections import deque
import numpy as np
import transforms3d as tr3d

class WalkerVelocitySmoother(WalkerWrapper):
    def __init__(self, robot: BaseWalker, queue_weights : np.ndarray = np.array([0.1,0.2,0.2,0.2,0.3])):
        super().__init__(robot)
        number_of_queues = len(queue_weights)
        self._velocity_queue = deque([],maxlen=number_of_queues)
        self._local_velocity_queue = deque([],maxlen=number_of_queues)
        self._queue_weights = queue_weights
        self._velocity_estimate = None
        self._local_velocity_estimate = None

    def update_velocity_estimate(self):
        if len(self._velocity_queue) == 0:
            self._velocity_estimate = np.zeros(3)
            self._local_velocity_estimate = np.zeros(3)
        else:
            accumulated_weight = 0.0
            self._velocity_estimate = np.zeros(3)
            self._local_velocity_estimate = np.zeros(3)
            length_of_current_queue = len(self._velocity_queue)
            for i in range(length_of_current_queue):
                weight = self._queue_weights[i + self._velocity_queue.maxlen - length_of_current_queue]
                self._velocity_estimate += weight * self._velocity_queue[i]
                self._local_velocity_estimate += weight * self._local_velocity_queue[i]
                accumulated_weight += weight
            self._velocity_estimate /= accumulated_weight
            self._local_velocity_estimate /= accumulated_weight

    def receive_observation(self) -> bool:
        ret = super().receive_observation()
        self._velocity_queue.append(self.robot.get_3d_linear_velocity())
        self._local_velocity_queue.append(self.robot.get_3d_local_velocity())
        self.update_velocity_estimate()
    
    def reset(self) -> None:
        super().reset()
        self._velocity_queue.clear()
        self._local_velocity_queue.clear()
        self._velocity_queue.append(self.robot.get_3d_linear_velocity())
        self._local_velocity_queue.append(self.robot.get_3d_local_velocity())
        self.update_velocity_estimate()
    
    def get_3d_linear_velocity(self) -> np.ndarray:
        return self._velocity_estimate

    def get_3d_local_velocity(self) -> np.ndarray:
        return self._local_velocity_estimate