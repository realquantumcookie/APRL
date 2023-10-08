import gym
import gym.spaces
from gym.core import Env
from jaxrl5.data.replay_buffer import ReplayBuffer
from rail_walker_interface import BaseWalker, JoystickPolicy, JoystickPolicyTargetProvider, JoystickPolicyTerminationConditionProvider, JoystickPolicyRewardProvider, JoystickEnvironment
import numpy as np
from typing import Any, Optional, List

from rail_walker_interface.joystick_policy.joystick_policy import JoystickPolicy

class RelabelTargetProvider(JoystickPolicyTargetProvider):
    def __init__(self, target_provider : JoystickPolicyTargetProvider, steps_every_new_target : int = 5):
        self.target_provider = target_provider
        self.steps_every_new_target = steps_every_new_target
        self._target_velocity = 0.0
        self._new_target_velocities : List[float] = []
        self._steps_until_new_target = 0

    def get_target_goal_world_delta(self, Robot: BaseWalker) -> np.ndarray:
        return self.target_provider.get_target_goal_world_delta(Robot)
    
    def get_target_velocity(self, Robot: BaseWalker) -> float:
        return self._target_velocity
    
    def is_target_velocity_fixed(self) -> bool:
        return False
    
    def has_target_changed(self) -> bool:
        return self._steps_until_new_target <= 0 or self.target_provider.has_target_changed()

    def step(
        self, 
        Robot: BaseWalker, 
        info_dict: dict[str,Any], 
        randomState : np.random.RandomState
    ) -> None:
        self._new_target_velocities.append(Robot.get_3d_local_velocity()[0])

    def after_step(
        self, 
        Robot: BaseWalker, 
        info_dict: dict[str,Any], 
        randomState : np.random.RandomState
    ) -> None:
        self._target_velocity = np.mean(self._new_target_velocities)
        self._steps_until_new_target = self.steps_every_new_target
        self._new_target_velocities.clear()

    def reset(
        self, 
        Robot: BaseWalker, 
        info_dict: dict[str,Any], 
        termination_provider_triggered: JoystickPolicyTerminationConditionProvider,
        randomState: np.random.RandomState
    ) -> None:
        self._steps_until_new_target = self.steps_every_new_target
        self._new_target_velocities.clear()

class RelabelAggregateWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, max_steps : int = 100_000, other_envs : list[gym.Env] = []):
        super().__init__(env)
        self.other_envs = other_envs
        for other_env in other_envs:
            assert hasattr(other_env, "joystick_policy"), "Other envs must have a joystick policy"
            other_joystick : JoystickPolicy = other_env.joystick_policy
            other_joystick.termination_providers.clear()

        self.prev_obs_list = []
        self._replay_buffer = ReplayBuffer(
            env.observation_space,
            env.action_space,
            capacity=max_steps,
        )
        self._enabled = True

    def enable_relabel(self):
        self._enabled = True

    def disable_relabel(self):
        self._enabled = False

    def get_relabel_replay_buffer(self) -> ReplayBuffer:
        return self._replay_buffer
    
    def set_relabel_replay_buffer(self, buffer : ReplayBuffer):
        self._replay_buffer = buffer
    
    def step(self, action):
        ret = super().step(action)
        
        if self._enabled:
            new_obs_list = []
            for idx, env in enumerate(self.other_envs):
                prev_obs = self.prev_obs_list[idx]
                obs, rew, done, info = env.step(action)
                done = done or ret[2]
                truncated = "TimeLimit.truncated" in info and info['TimeLimit.truncated']
                if (not done) or truncated:
                    mask = 1.0
                else:
                    mask = 0.0
                self._replay_buffer.insert(dict(
                    observations=prev_obs,
                    actions=action,
                    rewards=rew,
                    masks=mask,
                    dones=done,
                    next_observations=obs
                ))
                if 'TimeLimit.joystick_target_change' in info and info['TimeLimit.joystick_target_change']:
                    obs, info = env.reset(return_info=True)
                    done = False
            
                new_obs_list.append(obs)
            self.prev_obs_list = new_obs_list
        return ret
    
    def reset(self, *args, **kwargs):
        ret = super().reset(*args, **kwargs)
        self.prev_obs_list = [env.reset() for env in self.other_envs]
        return ret

class SymmetricRelabelWrapper(gym.Wrapper, JoystickEnvironment):
    def __init__(self, env: gym.Env, obs_space_before_flatten : gym.Space, max_steps : int = 100_000):
        super().__init__(env)
        self._replay_buffer = ReplayBuffer(
            env.observation_space,
            env.action_space,
            capacity=max_steps,
        )
        self.obs_space_before_flatten = obs_space_before_flatten
        self.prev_obs_relabeled = None
        self._enabled = True

    @property
    def joystick_policy(self) -> JoystickPolicy:
        return self.env.joystick_policy
    
    def set_joystick_policy(self, joystick_policy: JoystickPolicy):
        return self.env.set_joystrick_policy(joystick_policy)
    
    @property
    def is_resetter_policy(self) -> bool:
        return self.env.is_resetter_policy
    
    def relabel_action(self, action : np.ndarray):
        assert action.shape == (12,), "Action must be of shape (12,)"
        ret_action = np.zeros_like(action)
        ret_action[[0, 6]] = -action[[3, 9]]
        ret_action[[3, 9]] = -action[[0, 6]]
        ret_action[[1, 2, 7, 8]] = action[[4, 5, 10, 11]]
        ret_action[[4, 5, 10, 11]] = action[[1, 2, 7, 8]]
        return ret_action

    def relabel_obs(self, obs : np.ndarray) -> np.ndarray:
        ret_obs_unflattened = gym.spaces.unflatten(self.obs_space_before_flatten, obs)
        if "target_obs" in ret_obs_unflattened and ret_obs_unflattened["target_obs"] is not None:
            ret_obs_unflattened["target_obs"] = -ret_obs_unflattened["target_obs"] # Only supports relabeling TDY for now
        if "robot/joints_pos" in ret_obs_unflattened and ret_obs_unflattened["robot/joints_pos"] is not None:
            ret_obs_unflattened["robot/joints_pos"] = self.relabel_joint_potentially_stacked(ret_obs_unflattened["robot/joints_pos"])
        if "robot/joints_vel" in ret_obs_unflattened and ret_obs_unflattened["robot/joints_vel"] is not None:
            ret_obs_unflattened["robot/joints_vel"] = self.relabel_joint_potentially_stacked(ret_obs_unflattened["robot/joints_vel"])
        if "robot/torques" in ret_obs_unflattened and ret_obs_unflattened["robot/torques"] is not None:
            ret_obs_unflattened["robot/torques"] = self.relabel_joint_potentially_stacked(ret_obs_unflattened["robot/torques"])
        if "robot/foot_contacts" in ret_obs_unflattened and ret_obs_unflattened["robot/foot_contacts"] is not None:
            ret_obs_unflattened["robot/foot_contacts"] = self.relabel_feet_potentially_stacked(ret_obs_unflattened["robot/foot_contacts"])
        if "robot/foot_forces" in ret_obs_unflattened and ret_obs_unflattened["robot/foot_forces"] is not None:
            ret_obs_unflattened["robot/foot_forces"] = self.relabel_feet_potentially_stacked(ret_obs_unflattened["robot/foot_forces"])
        if "robot/foot_forces_normalized" in ret_obs_unflattened and ret_obs_unflattened["robot/foot_forces_normalized"] is not None:
            ret_obs_unflattened["robot/foot_forces_normalized"] = self.relabel_feet_potentially_stacked(ret_obs_unflattened["robot/foot_forces_normalized"])
        return gym.spaces.flatten(self.obs_space_before_flatten, ret_obs_unflattened)

    def relabel_feet_potentially_stacked(self, feet_obs : np.ndarray) -> np.ndarray:
        if feet_obs.ndim > 1:
            assert feet_obs.shape[-1] == 4
            return np.stack([
                self.relabel_feet_potentially_stacked(feet_obs[i]) for i in range(feet_obs.shape[0])
            ], axis=0)
        else:
            assert feet_obs.shape == (4,)
            ret_obs = np.zeros_like(feet_obs)
            ret_obs[[0, 2]] = feet_obs[[1, 3]]
            ret_obs[[1, 3]] = feet_obs[[0, 2]]
            return ret_obs

    def relabel_joint_potentially_stacked(self, joint_obs : np.ndarray) -> np.ndarray:
        if joint_obs.ndim > 1:
            assert joint_obs.shape[-1] == 12
            return np.stack([
                self.relabel_joint_potentially_stacked(joint_obs[i]) for i in range(joint_obs.shape[0])
            ], axis=0)
        else:
            return self.relabel_action(joint_obs)

    def enable_relabel(self):
        self._enabled = True

    def disable_relabel(self):
        self._enabled = False

    def get_relabel_replay_buffer(self) -> ReplayBuffer:
        return self._replay_buffer
    
    def set_relabel_replay_buffer(self, buffer : ReplayBuffer):
        self._replay_buffer = buffer
    
    def step(self, action):
        ret = super().step(action)
        
        if self._enabled:
            relabeled_action = self.relabel_action(action)
            relabeled_obs = self.relabel_obs(ret[0])
            truncated = "TimeLimit.truncated" in ret[3] and ret[3]['TimeLimit.truncated']
            self._replay_buffer.insert(dict(
                observations=self.prev_obs_relabeled,
                actions=relabeled_action,
                rewards=ret[1],
                masks=1.0 if not ret[2] or truncated else 0.0,
                dones=ret[3],
                next_observations=relabeled_obs
            ))
            self.prev_obs_relabeled = relabeled_obs
        return ret
    
    def reset(self, *args, **kwargs):
        ret = super().reset(*args, **kwargs)
        if kwargs.get("return_info", False):
            obs = ret[0]
        else:
            obs = ret
        self.prev_obs_relabeled = self.relabel_obs(obs)
        return ret