import gym
import gym.spaces
from .env import WalkerEnvironment, JoystickEnvironment
from ..robot import BaseWalker, BaseWalkerWithFootContact
from ..joystick_policy import JoystickPolicy
from functools import cached_property
import numpy as np
from typing import Optional, Any
import copy
from collections import OrderedDict

class JoystickEnvObservationExtractor:
    def __init__(self, env : "JoystickEnvImpl"):
        self.env = env

    @cached_property
    def observation_spec(self) -> gym.spaces.Dict:
        ret_dict = {
            "robot/joints_pos": gym.spaces.Box(
                low=self.env.robot.joint_qpos_mins,
                high=self.env.robot.joint_qpos_maxs,
                shape=(self.env.robot.joint_nums,),
                dtype=np.float32
            ),
            "robot/joints_vel": gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.env.robot.joint_nums,),
                dtype=np.float32
            ),
            "robot/imu": gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(4,),
            ),
            "robot/sensors_gyro": gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(3,),
                dtype=np.float32
            ),
            "robot/sensors_framequat": gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(4,),
                dtype=np.float32
            ),
            "robot/torques": gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.env.robot.joint_nums,),
                dtype=np.float32
            ),
            "robot/sensors_local_velocimeter": gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(3,),
                dtype=np.float32
            ),
            "robot/sensors_local_velocimeter_x": gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(1,),
                dtype=np.float32
            ),
            "robot/sensors_accelerometer": gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(3,),
                dtype=np.float32
            ),
        }
        if isinstance(self.env.robot.unwrapped(), BaseWalkerWithFootContact):
            ret_dict["robot/foot_forces"] = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(4,),
                dtype=np.float32
            )
            ret_dict["robot/foot_forces_normalized"] = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(4,),
                dtype=np.float32
            )
            ret_dict["robot/foot_forces_normalized_masked"] = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(4,),
                dtype=np.float32
            )
            ret_dict["robot/foot_contacts"] = gym.spaces.Box( # should use MultiBinary but flatten() does not support having multibinary / box spaces in a Dict
                low=0,
                high=1,
                shape=(4,),
                dtype=np.float32
            )
        return gym.spaces.Dict(ret_dict)
    
    def extract_observation(self) -> dict[str,Any]:
        roll, pitch, yaw = self.env.robot.get_roll_pitch_yaw()
        dr, dp, dy = self.env.robot.get_3d_angular_velocity()
        imu = np.array([roll, pitch, dr, dp], dtype=np.float32)

        ret_dict = {
            "robot/joints_pos": self.env.robot.get_joint_qpos(),
            "robot/joints_vel": self.env.robot.get_joint_qvel(),
            "robot/imu": imu,
            "robot/sensors_gyro": self.env.robot.get_3d_angular_velocity(),
            "robot/sensors_framequat": self.env.robot.get_framequat_wijk(),
            "robot/torques": self.env.robot.get_joint_torques(),
            "robot/sensors_local_velocimeter": self.env.robot.get_3d_local_velocity(),
            "robot/sensors_local_velocimeter_x": self.env.robot.get_3d_local_velocity()[0:1],
            "robot/sensors_accelerometer": self.env.robot.get_3d_acceleration_local(),
        }
        if isinstance(self.env.robot.unwrapped(), BaseWalkerWithFootContact):
            ret_dict["robot/foot_forces"] = self.env.robot.get_foot_force()
            ret_dict["robot/foot_contacts"] = self.env.robot.get_foot_contact()
            if hasattr(self.env.robot, "foot_contact_no_contact_threshold") and hasattr(self.env.robot, "foot_contact_has_contact_threshold"):
                ret_dict["robot/foot_forces_normalized"] = (ret_dict["robot/foot_forces"] - self.env.robot.foot_contact_no_contact_threshold) / (self.env.robot.foot_contact_has_contact_threshold - self.env.robot.foot_contact_no_contact_threshold)
            else:
                ret_dict["robot/foot_forces_normalized"] = ret_dict["robot/foot_forces"]
            masked_foot_forces = ret_dict["robot/foot_forces_normalized"].copy()
            masked_foot_forces[-1] = 0.0
            ret_dict["robot/foot_forces_normalized_masked"] = masked_foot_forces
        return ret_dict


class JoystickEnvImpl(gym.Env[dict[str,Any],np.ndarray], WalkerEnvironment, JoystickEnvironment):
    metadata = {
        "render_modes": []
    }

    def __init__(
            self, 
            joystick_policy : JoystickPolicy
        ):
        gym.Env.__init__(self)
        WalkerEnvironment.__init__(self)
        JoystickEnvironment.__init__(self)
        # ====================== Store Parameters ======================
        self._joystick_policy = joystick_policy
        self.obs_extractor = JoystickEnvObservationExtractor(self)
        self.random_state = np.random.RandomState()
    
    @property
    def action_space(self) -> gym.spaces.Box:
        return gym.spaces.Box(
            low=self.robot.action_qpos_mins,
            high=self.robot.action_qpos_maxs,
            dtype=np.float32
        )

    @property
    def observation_space(self) -> gym.spaces.Dict:
        robot_space = self.obs_extractor.observation_spec
        
        real_obs_space = {}
        for key, space in robot_space.items():
            if key.startswith("robot/") and key[len("robot/"):] in self.joystick_policy.enabled_observables:
                real_obs_space[key] = space        
        
        if self.joystick_policy.target_observable is not None:
            real_obs_space["target_obs"] = self.joystick_policy.target_observable.get_observation_spec()
        
        if not self.joystick_policy.target_yaw_provider.is_target_velocity_fixed():
            real_obs_space["target_vel"] = gym.spaces.Box(
                low = 0.0,
                high = np.inf,
                shape=(1,)
            )
        
        target_custom_data_spec = self.joystick_policy.target_yaw_provider.get_target_custom_data_observable_spec()
        if self.joystick_policy.enable_target_custom_obs and target_custom_data_spec is not None:
            real_obs_space["target_custom"] = target_custom_data_spec
        
        # Enforce order
        real_obs_space = OrderedDict(sorted(real_obs_space.items(), key=lambda t: t[0]))
        obs_space = gym.spaces.Dict(real_obs_space)
        return obs_space

    @property
    def joystick_policy(self) -> JoystickPolicy:
        return self._joystick_policy

    def set_joystick_policy(self, joystick_policy: JoystickPolicy):
        self._joystick_policy = joystick_policy

    @property
    def is_resetter_policy(self) -> bool:
        return False

    @property
    def robot(self) -> BaseWalker:
        return self.joystick_policy.robot
    
    def _get_obs(self) -> dict[str,np.ndarray]:
        robot_obs = self.obs_extractor.extract_observation()
        
        real_obs = {}
        for key, value in robot_obs.items():
            if key.startswith("robot/") and key[len("robot/"):] in self.joystick_policy.enabled_observables:
                real_obs[key] = value
        
        if self.joystick_policy.target_observable is not None:
            target_observable_obs = self.joystick_policy.target_observable.get_observation()
            real_obs["target_obs"] = target_observable_obs
        
        if not self.joystick_policy.target_yaw_provider.is_target_velocity_fixed():
            target_vel = self.joystick_policy._target_velocity
            real_obs["target_vel"] = np.array([target_vel], dtype=np.float32)

        target_custom_data_obs = self.joystick_policy.target_yaw_provider.get_target_custom_data_observable()
        if self.joystick_policy.enable_target_custom_obs and target_custom_data_obs is not None:
            real_obs["target_custom"] = target_custom_data_obs

        return real_obs

    def step(self, action : np.ndarray) -> tuple[dict[str,Any], float, bool, dict[str,Any]]:
        # ====================== Step Joystick Policy ======================
        self.joystick_policy.before_step(action, self.random_state)
        infos = self.joystick_policy.after_step(self.random_state)
        
        # ====================== Return ======================
        reward = self.joystick_policy.get_reward()
        obs = self._get_obs()
        truncated = self.joystick_policy.should_truncate()
        if truncated:
            infos["TimeLimit.truncated"] = True
        
        done = self.joystick_policy.should_terminate() or truncated
        
        return obs, reward, done, infos

    def reset(self, seed : int | None = None, options: dict[str, Any] | None = None, **kwargs) -> tuple[dict[str,Any], dict[str,Any]]:
        if seed is not None:
            self.random_state = np.random.RandomState(seed)
        
        infos = self.joystick_policy.reset(self.random_state)
        if kwargs.get("return_info", False):
            return self._get_obs(), infos
        else:
            return self._get_obs()

    def seed(self, seed : int | None = None) -> list[int]:
        self.random_state = np.random.RandomState(seed)

    def render(self, *args, **kwargs):
        raise NotImplementedError

    def close(self):
        self.robot.close()