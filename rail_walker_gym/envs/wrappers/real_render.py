from typing import Union
import gym
from rail_walker_interface import JoystickPolicy, JoystickEnvImpl
from rail_real_walker.robots import *
from rail_mujoco_walker import JoystickPolicyDMControlTask, Go1SimWalker, A1SimWalker
from ...joystick_policy import *
from dm_control.locomotion import arenas
from mujoco_utils import composer_utils
from dm_control import composer
from .dm_to_gym import DMControlMultiCameraRenderWrapper, RailWalkerMujocoComplianceWrapper
import numpy as np

class RealRenderWrapperWithSim(gym.Wrapper):
    def __init__(
        self,
        env : gym.Env
    ):
        assert isinstance(env.unwrapped, JoystickEnvImpl)
        super().__init__(env)
        self.init_sim_environment(256, 256)
    
    @property
    def real_env(self) -> JoystickEnvImpl:
        return self.env
    
    def init_sim_environment(self, render_height, render_width):
        assert isinstance(self.real_env.robot, Go1RealWalker)
        sim_robot = Go1SimWalker(
            Kp = self.real_env.robot.Kp,
            Kd = self.real_env.robot.Kd,
            action_interpolation=self.real_env.robot.action_interpolation,
            limit_action_range=self.real_env.robot.limit_action_range,
        )
        sim_robot.control_timestep = self.real_env.robot.control_timestep
        sim_robot.control_subtimestep = self.real_env.robot.control_subtimestep
        floor = arenas.Floor(
            size=(20.0, 20.0)
        )
        floor._top_camera.remove()
            
        sim_task = JoystickPolicyDMControlTask(
            joystick_policy=JoystickPolicy(
                robot=sim_robot,
                reward_provider=JoystickPolicyStrictRewardProvider(),
                target_yaw_provider=JoystickPolicyForwardOnlyTargetProvider(),
                termination_providers=[],
                truncation_providers=[],
                resetters=[],
                initializers=[],
                target_observable=None,
                enabled_observables=[]
            ),
            floor=floor
        )
        dm_env = composer_utils.Environment(
            task=sim_task,
            strip_singleton_obs_buffer_dim=True,
            recompile_physics=False,
        )
        # dm_env = composer.Environment(
        #     task=sim_task,
        #     strip_singleton_obs_buffer_dim=True
        # )
        sim_env = DMControlMultiCameraRenderWrapper(
            env=dm_env,
            scene_callback=sim_task.render_scene_callback,
            render_height=render_height,
            render_width=render_width,
        )
        sim_env = RailWalkerMujocoComplianceWrapper(
            sim_env,
        )
        self.sim_env = sim_env
    
    def align_sim_to_real(self):
        sim_robot : Go1SimWalker = self.sim_env.robot
        sim_policy : JoystickPolicy = self.sim_env.joystick_policy
        sim_task : JoystickPolicyDMControlTask = self.sim_env.task
        sim_robot.Kp = self.real_env.robot.Kp
        sim_robot.Kd = self.real_env.robot.Kd
        sim_robot.action_interpolation = self.real_env.robot.action_interpolation
        sim_robot.limit_action_range = self.real_env.robot.limit_action_range
        sim_robot.mujoco_walker._last_physics.bind(sim_robot.mujoco_walker.joints).qpos[:] = self.real_env.robot.get_joint_qpos()
        sim_robot.mujoco_walker.set_framequat(sim_robot.mujoco_walker._last_physics, self.real_env.robot.get_framequat_wijk())
        sim_policy._target_delta_yaw = self.real_env.joystick_policy.target_delta_yaw
        sim_policy._target_yaw = self.real_env.joystick_policy.target_yaw
        sim_policy._target_goal_world_delta = self.real_env.joystick_policy.target_goal_world_delta
        sim_policy._target_goal_local = self.real_env.joystick_policy.target_goal_local
        sim_robot.mujoco_walker.refresh_observation(sim_robot.mujoco_walker._last_physics)

    def step(self, action):
        ret = self.env.step(action)
        self.align_sim_to_real()
        return ret

    def reset(self, *args, **kwargs):
        ret = super().reset(**kwargs)
        self.sim_env.reset()
        self.align_sim_to_real()
        return ret
    
    def render(self, mode="human", **kwargs):
        return self.sim_env.render(mode=mode, **kwargs)

    @property
    def control_timestep(self):
        return self.real_env.robot.control_timestep