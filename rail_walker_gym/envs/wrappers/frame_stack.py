import gym
import gym.spaces
import collections
import copy
import numpy as np
from typing import List

class FrameStackWrapper(gym.ObservationWrapper):
    """Wrapper that stacks observations along a new final axis."""
    @staticmethod
    def stack_box_spec(spec : gym.spaces.Box, num_frames : int):
        low = np.expand_dims(spec.low, axis=-1).repeat(num_frames, axis=-1)
        high = np.expand_dims(spec.high, axis=-1).repeat(num_frames, axis=-1)
        new_shape = spec.shape + (num_frames,)
        return gym.spaces.Box(low, high, shape=new_shape, dtype=spec.dtype)

    @staticmethod
    def stack_multibinary_spec(spec: gym.spaces.MultiBinary, num_frames: int):
        new_spec = copy.copy(spec)
        if isinstance(spec.n, int):
            new_spec.n = (spec.n,num_frames)
        else:
            new_spec.n = spec.n + (num_frames,)
        return new_spec

    @staticmethod
    def stack_dict_spec(spec: gym.spaces.Dict, exclude_keys: List[str], num_frames: int):
        new_spec = copy.copy(spec)
        for key, value in spec.spaces.items():
            if key in exclude_keys:
                new_spec[key] = value
            elif isinstance(value, gym.spaces.Box):
                new_spec[key] = FrameStackWrapper.stack_box_spec(value, num_frames)
            elif isinstance(value, gym.spaces.Dict):
                new_spec[key] = FrameStackWrapper.stack_dict_spec(value, num_frames)
            elif isinstance(value, gym.spaces.MultiBinary):
                new_spec[key] = FrameStackWrapper.stack_multibinary_spec(value, num_frames)
            else:
                raise NotImplementedError
        return new_spec
    
    @staticmethod
    def stack_box_obs(all_obs : list[np.ndarray], num_frames : int):
        if len(all_obs) < num_frames:
            all_obs = [np.zeros_like(all_obs[0])]*(num_frames - len(all_obs)) + all_obs
        return np.stack(all_obs, axis=-1)
    
    @staticmethod
    def stack_multibinary_obs(all_obs : list[np.ndarray], num_frames : int):
        return __class__.stack_box_obs(all_obs, num_frames)

    @staticmethod
    def stack_dict_obs(all_obs : list[dict[str,np.ndarray]], exclude_keys: List[str], num_frames : int):
        new_obs = {}
        for key, value in all_obs[-1].items():
            if key in exclude_keys:
                new_obs[key] = value
            elif isinstance(value,dict) or isinstance(value, collections.OrderedDict):
                new_obs[key] = FrameStackWrapper.stack_dict_obs(
                    [obs[key] for obs in all_obs],
                    num_frames
                )
            elif isinstance(value, np.ndarray):
                if value.dtype == np.int8:
                    new_obs[key] = FrameStackWrapper.stack_multibinary_obs(
                        [obs[key] for obs in all_obs],
                        num_frames
                    )
                else:
                    new_obs[key] = FrameStackWrapper.stack_box_obs(
                        [obs[key] for obs in all_obs],
                        num_frames
                    )
            else:
                raise NotImplementedError
        return new_obs

    def __init__(
        self,
        env: gym.Env,
        num_frames: int = 4,
        exclude_keys: List[str] = [],
    ):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Dict) or isinstance(env.observation_space, gym.spaces.Box) or isinstance(env.observation_space, gym.spaces.MultiBinary), "FrameStackWrapper only works with Dict, Box, or MultiBinary observation spaces"
        assert isinstance(env.observation_space, gym.spaces.Dict) or len(exclude_keys) == 0, "FrameStackWrapper does not support excluding keys from Dict observation spaces, {}".format(env.observation_space)

        self.previous_queue = collections.deque([],maxlen=num_frames)
        if isinstance(env.observation_space, gym.spaces.Dict):
            self.observation_space = __class__.stack_dict_spec(env.observation_space, exclude_keys, num_frames)
        elif isinstance(env.observation_space, gym.spaces.Box):
            self.observation_space = __class__.stack_box_spec(env.observation_space, num_frames)
        elif isinstance(env.observation_space, gym.spaces.MultiBinary):
            self.observation_space = __class__.stack_multibinary_spec(env.observation_space, num_frames)
        else:
            raise NotImplementedError
        self._last_joystick_truncation = False
        self.exclude_keys = exclude_keys

    def reset(self, *args, **kwargs):
        resetret = self.env.reset(*args, **kwargs)
        if self._last_joystick_truncation:
            self.previous_queue.pop()
        else:
            self.previous_queue.clear()
        self._last_joystick_truncation = False
        if kwargs.get("return_info", False):
            obs, info = resetret
            self.previous_queue.append(obs)
            return self.observation(obs), info
        else:
            obs = resetret
            self.previous_queue.append(obs)
            return self.observation(obs)

    def step(self, action):
        obs, rew, done, info  = self.env.step(action)
        if done and "TimeLimit.joystick_target_change" in info and info["TimeLimit.joystick_target_change"]:
            self._last_joystick_truncation = True
        else:
            self._last_joystick_truncation = False
        
        self.previous_queue.append(obs)
        return self.observation(obs), rew, done, info

    @property
    def num_frames(self):
        return self.previous_queue.maxlen

    def observation(self, observation):
        if isinstance(observation, dict) or isinstance(observation, collections.OrderedDict):
            new_obs = __class__.stack_dict_obs(self.previous_queue, self.exclude_keys, self.num_frames)
        elif isinstance(observation, np.ndarray):
            if observation.dtype == np.int8:
                new_obs = __class__.stack_multibinary_obs(self.previous_queue, self.num_frames)
            else:
                new_obs = __class__.stack_box_obs(list(self.previous_queue), self.num_frames)
        return new_obs