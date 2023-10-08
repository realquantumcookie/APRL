import gym
import gym.spaces
from typing import Optional, Union
import numpy as np

class RescaleActionAsymmetric(gym.ActionWrapper):
    def __init__(
        self, 
        env : gym.Env, 
        low : Union[float, np.ndarray],
        high : Union[float, np.ndarray],
        center_action : Optional[np.ndarray] = None
    ):
        super().__init__(env)
        self._center_action = center_action
        self.low = low
        self.high = high
    
    @property
    def center_action(self) -> np.ndarray:
        return self._center_action if self._center_action is not None else (
            (self.env.action_space.high + self.env.action_space.low) / 2.0
        )
    
    @center_action.setter
    def center_action(self, value : Optional[np.ndarray]):
        self._center_action = value
    
    @property
    def action_space(self) -> gym.spaces.Box:
        return gym.spaces.Box(
            low = self.low,
            high = self.high,
            shape=self.env.action_space.shape,
            dtype=self.env.action_space.dtype
        )

    def action(self, action : np.ndarray):
        new_center = (self.high + self.low) / 2.0
        
        new_delta_action = action - new_center
        below_center_idx = new_delta_action < 0
        other_idx = np.logical_not(below_center_idx)

        new_delta_high = self.high - new_center
        if not isinstance(new_delta_high, float):
            new_delta_high = new_delta_high[other_idx]
        new_delta_low = new_center - self.low
        if not isinstance(new_delta_low, float):
            new_delta_low = new_delta_low[below_center_idx]

        center_action = self.center_action

        delta_center = new_delta_action.copy()
        delta_center[below_center_idx] *= (center_action[below_center_idx] - self.env.action_space.low[below_center_idx]) / new_delta_low
        delta_center[other_idx] *= (self.env.action_space.high[other_idx] - center_action[other_idx]) / new_delta_high
        ret = (center_action + delta_center).astype(self.env.action_space.dtype)
        # print(action, "=>", ret)
        # print(self.env.action_space)
        return ret

    def reverse_action(self, action : np.ndarray):
        delta_center = action - self.center_action
        below_center_idx = delta_center < 0
        other_idx = np.logical_not(below_center_idx)

        new_center = (self.high + self.low) / 2.0
        new_delta_high = self.high - new_center
        if not isinstance(new_delta_high, float):
            new_delta_high = new_delta_high[other_idx]
        new_delta_low = new_center - self.low
        if not isinstance(new_delta_low, float):
            new_delta_low = new_delta_low[below_center_idx]

        center_action = self.center_action

        new_delta_center = delta_center.copy()
        new_delta_center[below_center_idx] *= new_delta_low / (center_action[below_center_idx] - self.env.action_space.low[below_center_idx])
        new_delta_center[other_idx] *= new_delta_high / (self.env.action_space.high[other_idx] - center_action[other_idx])
        return new_center + new_delta_center
