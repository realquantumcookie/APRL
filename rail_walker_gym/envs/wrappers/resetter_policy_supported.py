import gym
import gym.spaces
from rail_walker_interface import JoystickEnvironment, JoystickPolicy, JoystickPolicyResetter
from rail_mujoco_walker import JoystickPolicyDMControlTask
from jaxrl5.agents.agent import Agent as JaxAgent
import typing
import numpy as np
from dmcgym.env import DMCGYM
from rail_walker_interface import JoystickEnvImpl


def resetter_rescale_action(action, old_min, old_max, new_min, new_max):
    return (action - old_min) / (old_max - old_min) * (new_max - new_min) + new_min

class ResetterPolicySupportedEnvironment(gym.Wrapper, JoystickEnvironment):
    @staticmethod
    def compute_obs_act_space_reset_agent(
        env: typing.Union[gym.Env,JoystickEnvironment],
        resetter_joystick_policy: JoystickPolicy,
        wrap_resetter_env_lambda : typing.Callable[[gym.Env], gym.Env] = lambda env: env,
        resetter_action_scale : typing.Optional[float] = 1.0,
    ):
        old_joy_policy = env.joystick_policy
        old_action_scale = old_joy_policy.robot.limit_action_range

        if resetter_action_scale is not None:
            old_joy_policy.robot.limit_action_range = resetter_action_scale
        env.set_joystick_policy(resetter_joystick_policy)
        
        reset_env = wrap_resetter_env_lambda(env)
        obs_space, act_space = reset_env.observation_space, reset_env.action_space
        
        old_joy_policy.robot.limit_action_range = old_action_scale
        env.set_joystick_policy(old_joy_policy)
        return obs_space, act_space

    def __init__(
        self, 
        env: typing.Union[gym.Env,JoystickEnvironment],
        resetter_joystick_policy: JoystickPolicy,
        resetter_agent: JaxAgent,
        max_seconds : float = 5.0,
        intercept_all_reset_calls = True,
        wrap_resetter_env_lambda : typing.Callable[[gym.Env], gym.Env] = lambda env: env,
        fallback_resetter: typing.Optional[JoystickPolicyResetter] = None,
        max_trials : int = 1,
        resetter_Kp : typing.Optional[float] = None,
        resetter_Kd : typing.Optional[float] = None,
        resetter_action_interpolation : typing.Optional[bool] = None,
        resetter_action_scale : typing.Optional[float] = None,
        resetter_power_protect_factor : typing.Optional[float] = None,
    ):
        assert hasattr(env, "joystick_policy"), "The environment must have a joystick policy"
        assert hasattr(env, "set_joystick_policy"), "The environment must have a set_joystick_policy method"
        
        gym.Wrapper.__init__(
            self, env
        )
        JoystickEnvironment.__init__(self)
        self.resetter_policy_joystick_policy = resetter_joystick_policy
        self.resetter_policy_agent = resetter_agent
        self.resetter_policy_max_seconds = max_seconds
        self.resetter_env = None
        self.resetter_Kp = resetter_Kp
        self.resetter_Kd = resetter_Kd
        self.resetter_action_interpolation = resetter_action_interpolation
        self.resetter_action_scale = resetter_action_scale
        self.resetter_power_protect_factor = resetter_power_protect_factor

        self.first_reset = True
        self.wrap_resetter_env_lambda = wrap_resetter_env_lambda
        self.random_state = np.random.RandomState(0)
        self.intercept_all_reset_calls = intercept_all_reset_calls

        self.max_trials = max_trials
        self.fallback_resetter = fallback_resetter

        self.replace_cache = None
        
        # Hacking the render method
        self.prepended_frames = None

    def replace_resetter_settings(self):
        assert self.replace_cache is None, "The replace cache is already set"
        self.replace_cache = {}
        if self.resetter_Kp is not None:
            self.replace_cache["Kp"] = self.joystick_policy.robot.Kp
            self.joystick_policy.robot.Kp = self.resetter_Kp
        if self.resetter_Kd is not None:
            self.replace_cache["Kd"] = self.joystick_policy.robot.Kd
            self.joystick_policy.robot.Kd = self.resetter_Kd
        if self.resetter_action_interpolation is not None:
            self.replace_cache["action_interpolation"] = self.joystick_policy.robot.action_interpolation
            self.joystick_policy.robot.action_interpolation = self.resetter_action_interpolation
        if self.resetter_action_scale is not None:
            self.replace_cache["action_scale"] = self.joystick_policy.robot.limit_action_range
            self.joystick_policy.robot.limit_action_range = self.resetter_action_scale
        if self.resetter_power_protect_factor is not None:
            self.replace_cache["power_protect_factor"] = self.joystick_policy.robot.power_protect_factor
            self.joystick_policy.robot.power_protect_factor = self.resetter_power_protect_factor
        
        self.replace_cache["joystick_policy"] = self.joystick_policy
        self.set_joystick_policy(self.resetter_policy_joystick_policy)

    def recover_original_settings(self):
        assert self.replace_cache is not None, "The replace cache is not set"
        if "Kp" in self.replace_cache:
            self.joystick_policy.robot.Kp = self.replace_cache["Kp"]
        if "Kd" in self.replace_cache:
            self.joystick_policy.robot.Kd = self.replace_cache["Kd"]
        if "action_interpolation" in self.replace_cache:
            self.joystick_policy.robot.action_interpolation = self.replace_cache["action_interpolation"]
        if "action_scale" in self.replace_cache:
            self.joystick_policy.robot.limit_action_range = self.replace_cache["action_scale"]
        if "power_protect_factor" in self.replace_cache:
            self.joystick_policy.robot.power_protect_factor = self.replace_cache["power_protect_factor"]
        self.set_joystick_policy(self.replace_cache["joystick_policy"])
        self.replace_cache = None

    @property
    def intercept_all_reset_calls(self):
        return self._intercept_all_reset_calls
    
    @intercept_all_reset_calls.setter
    def intercept_all_reset_calls(self, value):
        self._intercept_all_reset_calls = value
        self._inject_mujoco()
    
    def _inject_mujoco(self):
        if isinstance(self.env.unwrapped, DMCGYM) and isinstance(self.env.unwrapped._env.task,JoystickPolicyDMControlTask):
            if self._intercept_all_reset_calls:
                self.env.unwrapped._env.task.should_terminate_episode = lambda *args, **kwargs: False
            else:
                self.env.unwrapped._env.task.should_terminate_episode = JoystickPolicyDMControlTask.should_terminate_episode

    def _init_resetter_env(self):
        self.replace_resetter_settings()
        print("=================== Initializing reset policy environment ===================")
        print("Resetter Policy Environment Action Space Before Wrapping", self.env.action_space)
        tmp_resetter_env = gym.Wrapper(self.env)
        # Intercept the reset call to resetter_env
        def new_reset(*args, **kwargs):
            self.resetter_policy_joystick_policy.reset(
                self.random_state
            )
            if kwargs.get("return_info",False):
                ret = self.receive_observation(tmp_resetter_env)[:2]
            else:
                ret = self.receive_observation(tmp_resetter_env)[0]
            return ret
        tmp_resetter_env.reset = new_reset
        
        resetter_env = self.wrap_resetter_env_lambda(tmp_resetter_env)
        obs = resetter_env.reset(seed=0)
        self.resetter_policy_agent.eval_actions(obs) # Jit the resetter policy
        self.resetter_env = resetter_env
        self.recover_original_settings()
        print("=================== Initialized reset policy environment ===================")

    def receive_observation(self,env:gym.Env):
        # if isinstance(self.env, DMCGYM) and isinstance(self.env._env, dmcenv.Environment):
        #     dm_control_env = self.env._env
        #     dm_control_env._observation_updater.update()
        #     obs = dmc_obs2gym_obs(dm_control_env._observation_updater.get_observation())
        

        if isinstance(self.env,JoystickEnvImpl):
            obs = self.env._get_obs()
            done = self.env.joystick_policy.should_terminate() or self.env.joystick_policy.should_truncate()
            info = self.env.joystick_policy.last_info
            return obs, info, done

        last_action = self.joystick_policy.robot.get_joint_qpos()
        rescaled_action = resetter_rescale_action(
            last_action,
            self.joystick_policy.robot.action_qpos_mins, #env.unwrapped.action_space.low,
            self.joystick_policy.robot.action_qpos_maxs, #env.unwrapped.action_space.high,
            env.action_space.low,
            env.action_space.high
        )
        rescaled_action = np.clip(rescaled_action.astype(env.action_space.dtype), env.action_space.low, env.action_space.high)
        obs, _, done, info = env.step(rescaled_action)
        return obs, info, done

    def perform_resetter_policy(self):
        print("====================== Resetter Policy ======================")
        print("Replacing the joystick policy with resetter policy")
        is_recording = hasattr(self.env.unwrapped, "_is_video_rec") and self.env.unwrapped._is_video_rec
        recorded_frames = []
        self.replace_resetter_settings()
        obs, info = self.resetter_env.reset(return_info=True)
        done = self.resetter_policy_joystick_policy.should_terminate() or self.resetter_policy_joystick_policy.should_truncate()

        self.resetter_policy_joystick_policy.reset(self.random_state)
        if is_recording:
            recorded_frames.append(self.env.render(mode="rgb_array"))

        max_steps = int(self.resetter_policy_max_seconds / self.joystick_policy.control_timestep)
        
        for _ in range(max_steps):
            if done:
                break
            action = self.resetter_policy_agent.eval_actions(obs)
            next_obs, reward, done, info = self.resetter_env.step(action)
            if is_recording:
                recorded_frames.append(self.env.render(mode="rgb_array"))
            obs = next_obs
        
        print("Resetter policy done")
        self.recover_original_settings()
        if is_recording:
            self.prepended_frames = recorded_frames
        print("=============================================================")

    def reset(self, seed : int | None = None, *args, **kwargs):
        # before resetting the environment, we first perform action based on resetter policy
        # and then reset the environment
        if seed is not None:
            self.random_state = np.random.RandomState(seed)
        
        if self.first_reset:
            ret = self.env.reset(*args,seed=seed,**kwargs)
            self._init_resetter_env()
            self.first_reset = False
        else:
            if not self.intercept_all_reset_calls:
                self.perform_resetter_policy()
                ret = self.env.reset(*args,seed=seed,**kwargs)
            else:
                done = True
                
                for _ in range(self.max_trials):
                    if not done:
                        break
                    self.perform_resetter_policy()
                    # When receiving observation, we might be stepping to cause the episode to terminate
                    # In that case, we need to loop resetting the environment until we get a valid observation
                    joystick_policy : JoystickPolicy = self.joystick_policy
                    joystick_policy.reset(self.random_state)
                    obs, info, done = self.receive_observation(self.env)
                
                if done and self.fallback_resetter is not None:
                    self.fallback_resetter.reset(self.joystick_policy.robot, self.joystick_policy._info_dict, self.joystick_policy._termination_reason, self.random_state)
                    joystick_policy.reset(self.random_state)
                    obs, info, done = self.receive_observation(self.env)
                    if done:
                        raise RuntimeError("Fallback resetter failed")

                if kwargs.get("return_info", False):
                    ret = obs, {}
                else:
                    ret = obs
        return ret

    @property
    def observation_space(self) -> gym.Space:
        return super().observation_space
    
    @property
    def action_space(self) -> gym.Space:
        return super().action_space

    @property
    def joystick_policy(self) -> JoystickPolicy:
        return self.env.joystick_policy

    def set_joystick_policy(self, joystick_policy : JoystickPolicy):
        self.env.set_joystick_policy(joystick_policy)

    @property
    def is_resetter_policy(self) -> bool:
        return True