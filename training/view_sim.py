import dmcgym
import gym
from dm_control import viewer

#import rail_walker_gym.envs.register_real
import rail_walker_gym.envs.register_mujoco
from rail_mujoco_walker import JoystickPolicyDMControlTask
from rail_walker_gym import \
    JoystickPolicyRouteFollow2DTraversibleTargetProvider

env = gym.make("Go1GoalGraphFullSquishedMujoco-TDY-StrictRew-v0")
env.reset()

if isinstance(env.unwrapped, dmcgym.DMCGYM):
    task : JoystickPolicyDMControlTask = env.unwrapped._env.task
    if isinstance(task.joystick_policy.target_yaw_provider, JoystickPolicyRouteFollow2DTraversibleTargetProvider):
        traversible = task.joystick_policy.target_yaw_provider.traversible
        if hasattr(traversible, "add_sim_renderables"):
            traversible.add_sim_renderables(
                task._floor._mjcf_root,
                lambda pos: 0.5, #task._floor.height_lookup,
                render_height_offset=0.0,
                show_edges=True
            )

viewer.launch(env.unwrapped._env)
env.close()