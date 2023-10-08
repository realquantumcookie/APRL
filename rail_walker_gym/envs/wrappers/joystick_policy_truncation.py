import gym
from gym.core import Env
import numpy as np
from rail_walker_interface import JoystickEnvironment, JoystickPolicy
from rail_walker_interface.joystick_policy.joystick_policy import JoystickPolicy

class JoystickPolicyTruncationWrapper(gym.Wrapper, JoystickEnvironment):
    def __init__(self, env: Env):
        assert hasattr(env, "joystick_policy") and isinstance(env.joystick_policy, JoystickPolicy)
        gym.Wrapper.__init__(
            self, env
        )
        JoystickEnvironment.__init__(self)
        self._last_obs = None
        self.random_state = np.random.RandomState()
    
    @property
    def joystick_policy(self) -> JoystickPolicy:
        return self.env.joystick_policy
    
    def set_joystick_policy(self, joystick_policy: JoystickPolicy):
        return self.env.set_joystick_policy(joystick_policy)
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if not done and (
            self.joystick_policy.lock_target and self.joystick_policy.has_after_after_step
        ):
            done = True
            info['TimeLimit.truncated'] = True
            info['TimeLimit.joystick_target_change'] = True
            self._last_obs = obs
        else:
            self._last_obs = None
        return obs, reward, done, info
    
    def seed(self, seed=None):
        self.random_state = np.random.RandomState(seed)
        return self.env.seed(seed)
    
    def reset(self, **kwargs):
        if self._last_obs is not None:
            assert self.joystick_policy.has_after_after_step
            self.joystick_policy.after_after_step(self.random_state)
            if self.joystick_policy.target_observable is not None:
                self._last_obs["target_obs"] = self.joystick_policy.target_observable.get_observation()
            target_obs = self.joystick_policy.target_yaw_provider.get_target_custom_data_observable()
            if self.joystick_policy.enable_target_custom_obs and target_obs is not None:
                self._last_obs["target_custom"] = target_obs
            obs = self._last_obs
            self._last_obs = None
            if kwargs.get("return_info", False):
                return obs, self.joystick_policy.last_info
            else:
                return obs
            
        else:
            return self.env.reset(**kwargs)