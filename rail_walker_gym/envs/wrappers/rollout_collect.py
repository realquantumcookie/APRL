import numpy as np
import gym
from typing import Any, Tuple, List

class Rollout:
    def __init__(self) -> None:
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []

    def append(self, observation : np.ndarray, action : np.ndarray, reward : float, done : bool) -> None:
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
    
    def current_episode(self) -> Tuple[np.ndarray, np.ndarray, float, bool]:
        last_done = len(self.dones) - 1
        last_start = last_done
        while last_start > 0 and not self.dones[last_start - 1]:
            last_start -= 1
        
        return np.array(self.observations[last_start:last_done+1]), np.array(self.actions[last_start:last_done+1]), np.array(self.rewards[last_start:last_done+1]), np.array(self.dones[last_start:last_done+1])

    def last_episode(self) -> Tuple[List[np.ndarray], List[np.ndarray], List[float], List[bool]]:
        # Find the last done
        last_done = len(self.dones) - 1
        while last_done > 0 and not self.dones[last_done]:
            last_done -= 1

        last_start = last_done
        while last_start > 0 and not self.dones[last_start - 1]:
            last_start -= 1
        
        # Return the last episode
        return np.array(self.observations[last_start:last_done+1]), np.array(self.actions[last_start:last_done+1]), np.array(self.rewards[last_start:last_done+1]), np.array(self.dones[last_start:last_done+1])

    def export_npz(self, path : str) -> None:
        np.savez_compressed(path, observations=self.observations, actions=self.actions, rewards=self.rewards, dones=self.dones)
    
    def export_last_episode_npz(self, path : str) -> None:
        observations, actions, rewards, dones = self.last_episode()
        np.savez_compressed(path, observations=observations, actions=actions, rewards=rewards, dones=dones)
    
    def export_current_episode_npz(self, path : str) -> None:
        observations, actions, rewards, dones = self.current_episode()
        np.savez_compressed(path, observations=observations, actions=actions, rewards=rewards, dones=dones)

    def is_current_episode_empty(self) -> bool:
        return len(self.dones) == 0 or self.dones[-1]

    def clear(self) -> None:
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()

    @staticmethod
    def import_npz(path : str):
        roll = Rollout()
        data = np.load(path)
        roll.observations = list(data["observations"])
        roll.actions = list(data["actions"])
        roll.rewards = list(data["rewards"])
        roll.dones = list(data["dones"])
        return roll
    
    def __len__(self) -> int:
        return len(self.dones)

class RolloutCollector(gym.Wrapper):
    def __init__(self, env : gym.Env) -> None:
        super().__init__(env)
        self.collected_rollouts = Rollout()
        self.next_observation = None

    def step(self, action) -> Tuple[Any, float, bool, dict]:
        ret = self.env.step(action)
        self.collected_rollouts.append(self.next_observation, action, ret[1], ret[2])
        self.next_observation = ret[0]
        return ret
    
    def reset(self, *args, **kwargs):
        ret = self.env.reset(*args,**kwargs)
        if isinstance(ret, tuple) and isinstance(ret[1], dict):
            self.next_observation = ret[0]
        else:
            self.next_observation = ret
        return ret




