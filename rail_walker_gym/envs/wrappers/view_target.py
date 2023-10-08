from typing import Union, Any
import gym
from rail_walker_interface import BaseWalker, JoystickPolicy, JoystickEnvironment
import numpy as np
import matplotlib.pyplot as plt
from rail_walker_interface.joystick_policy.joystick_policy import JoystickPolicy

class JoystickTargetViewer(gym.Wrapper, JoystickEnvironment):
    def __init__(self, env: gym.Env):
        assert hasattr(env, 'joystick_policy')
        gym.Wrapper.__init__(self, env)
        JoystickEnvironment.__init__(self)
        plt.ion()
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        self.heading_arrow = plt.arrow(0, 0, 0, 1, color='g', width=0.1)
        self.tdy_arrow = plt.arrow(0, 0, 0, 1, color='y', width=0.1)

    @property
    def joystick_policy(self) -> JoystickPolicy:
        return self.env.joystick_policy
    
    def set_joystick_policy(self, joystick_policy: JoystickPolicy):
        return self.env.set_joystick_policy(joystick_policy)
    
    def step(self, action):
        ret = self.env.step(action)
        self.update()
        return ret
    
    def reset(self, **kwargs) -> Any | tuple[Any, dict]:
        ret = super().reset(**kwargs)
        self.update()
        return ret

    def update(self):
        heading_yaw = self.joystick_policy.robot.get_roll_pitch_yaw()[2]
        self.heading_arrow.set_data(
            x=0,
            y=0,
            dx=np.cos(heading_yaw),
            dy=np.sin(heading_yaw)
        )
        if hasattr(self.joystick_policy.reward_provider, 'target_linear_velocity'):
            target_velocity = self.joystick_policy.reward_provider.target_linear_velocity
        else:
            target_velocity = 1.0
        
        ty_with_tv = self.joystick_policy.target_goal_world_delta_unit * target_velocity
        self.tdy_arrow.set_data(
            x=0,
            y=0,
            dx=ty_with_tv[0],
            dy=ty_with_tv[1]
        )
        plt.pause(0.001)