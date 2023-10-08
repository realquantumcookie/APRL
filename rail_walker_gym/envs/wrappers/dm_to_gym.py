from typing import Callable, Optional, Any
import numpy as np
import math
import gym
import gym.spaces
import dmcgym
import dmcgym.env
from dm_control import composer
from dm_control.rl import control
from dm_control.mujoco.engine import Physics as EnginePhysics
from rail_walker_interface import JoystickEnvironment, WalkerEnvironment, BaseWalker, JoystickPolicy
from rail_mujoco_walker import JoystickPolicyDMControlTask
import mujoco
import dm_env
from collections import OrderedDict

class RailWalkerMujocoComplianceWrapper(gym.ObservationWrapper, JoystickEnvironment, WalkerEnvironment):
    def __init__(self, env: gym.Env):
        assert isinstance(env.unwrapped, dmcgym.DMCGYM)
        assert isinstance(env.unwrapped._env.task, JoystickPolicyDMControlTask)
        gym.Wrapper.__init__(
            self, env
        )
        JoystickEnvironment.__init__(self)
        WalkerEnvironment.__init__(self)
    
    @property
    def observation_space(self) -> gym.Space:
        robot_space = self.env.observation_space
        real_obs_space = {}
        for key, space in robot_space.items():
            if key.startswith("robot/") and key[len("robot/"):] in self.joystick_policy.enabled_observables:
                real_obs_space[key] = space        
        
        if self.joystick_policy.target_observable is not None:
            real_obs_space["target_obs"] = self.joystick_policy.target_observable.get_observation_spec()

        target_custom_data_spec = self.joystick_policy.target_yaw_provider.get_target_custom_data_observable_spec()
        if self.joystick_policy.enable_target_custom_obs and target_custom_data_spec is not None:
            real_obs_space["target_custom"] = target_custom_data_spec
        
        if not self.joystick_policy.target_yaw_provider.is_target_velocity_fixed():
            real_obs_space["target_vel"] = gym.spaces.Box(
                low = 0.0,
                high = np.inf,
                shape=(1,)
            )
        
        # Enforce order
        real_obs_space = OrderedDict(sorted(real_obs_space.items(), key=lambda t: t[0]))
        obs_space = gym.spaces.Dict(real_obs_space)
        return obs_space

    @property
    def action_space(self) -> gym.Space:
        return gym.spaces.Box(
            low = self.robot.action_qpos_mins,
            high = self.robot.action_qpos_maxs,
            dtype = np.float32
        )

    @property
    def robot(self) -> BaseWalker:
        task : JoystickPolicyDMControlTask = self.env.unwrapped._env.task
        return task.robot

    @property
    def joystick_policy(self) -> JoystickPolicy:
        task : JoystickPolicyDMControlTask = self.env.unwrapped._env.task
        return task.joystick_policy
    
    def set_joystick_policy(self, joystick_policy: JoystickPolicy):
        task : JoystickPolicyDMControlTask = self.env.unwrapped._env.task
        task.joystick_policy = joystick_policy
    
    @property
    def is_resetter_policy(self) -> bool:
        return False

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        info.update(self.joystick_policy.last_info)
        done = done or self.joystick_policy.should_terminate() or self.joystick_policy.should_truncate()
        if self.joystick_policy.should_truncate():
            info["TimeLimit.truncated"] = True
        return self.observation(obs), reward, done, info

    def observation(self, observation):
        obs = {}
        for key, value in observation.items():
            if key.startswith("robot/") and key[len("robot/"):] in self.joystick_policy.enabled_observables:
                obs[key] = value
        if self.joystick_policy.target_observable is not None:
            obs["target_obs"] = self.joystick_policy.target_observable.get_observation()
        
        if not self.joystick_policy.target_yaw_provider.is_target_velocity_fixed():
            target_vel = self.joystick_policy._target_velocity
            obs["target_vel"] = np.array([target_vel])
        
        target_custom_data_obs = self.joystick_policy.target_yaw_provider.get_target_custom_data_observable()
        if self.joystick_policy.enable_target_custom_obs and target_custom_data_obs is not None:
            obs["target_custom"] = target_custom_data_obs
        return obs

class DMControlMultiCameraRenderWrapper(dmcgym.DMCGYM):    
    def __init__(
        self,
        env: composer.Environment | control.Environment | dm_env.Environment,
        scene_callback : Optional[Callable[[EnginePhysics, mujoco.MjvScene], None]] = None,
        render_height: int = 84,
        render_width: int = 84,
    ):
        self.metadata["render_modes"].append("multi_camera")
        super().__init__(
            env
        )
        self.render_height = render_height
        self.render_width = render_width
        self.scene_callback = scene_callback

    def render(
        self,
        *args,
        **kwargs
    ) -> np.ndarray | None:
        return self._internal_render(
            height=self.render_height,
            width=self.render_width,
            scene_callback=self.scene_callback
        )
    
    def reset(self, seed : int | None = None, options: dict[str, Any] | None = None, **kwargs):
        if seed is not None:
            self.seed(seed)
        obs = super().reset()
        if kwargs.get("return_info",False):
            return obs, {}
        else:
            return obs

    def _internal_render(
        self,
        height: int = 84,
        width: int = 84,
        scene_callback = None
    ):
        
        physics = self._env.physics
        num_cameras = physics.model.ncam
        num_columns = int(math.ceil(math.sqrt(num_cameras)))
        num_rows = int(math.ceil(float(num_cameras) / num_columns))
        frame = np.zeros((num_rows * height, num_columns * width, 3), dtype=np.uint8)
        for col in range(num_columns):
            for row in range(num_rows):
                camera_id = row * num_columns + col
                if camera_id >= num_cameras:
                    break
                subframe = physics.render(
                    camera_id=camera_id, height=height, width=width, scene_callback=scene_callback
                )
                frame[
                    row * height : (row + 1) * height, col * width : (col + 1) * width
                ] = subframe
        return frame

