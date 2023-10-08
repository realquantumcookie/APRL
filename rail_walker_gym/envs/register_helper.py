import typing
import gym
import gym.wrappers
from gym.envs.registration import register
from typing import Callable
from .wrappers import *
from rail_walker_interface import BaseWalker, JoystickPolicy
from jaxrl5.wrappers import wrap_gym
# from jaxrl5.wrappers.single_precision import SinglePrecision
# from jaxrl5.wrappers.universal_seed import UniversalSeed

# def wrap_gym(env: gym.Env, rescale_actions: bool = True) -> gym.Env:
#     env = SinglePrecision(env)
#     env = UniversalSeed(env)
#     if rescale_actions:
#         env = gym.wrappers.RescaleAction(env, -1, 1)

#     env = gym.wrappers.ClipAction(env)

#     return env

def common_post_process_wrapper(env : gym.Env, action_range: float, filter_actions : int = 0, frame_stack : int = 3, action_history : int = 3, center_init_action : bool = False, log_print = False):
    assert hasattr(env, "joystick_policy"), "The environment must have a joystick policy"
    joystick_policy : JoystickPolicy = env.joystick_policy
    robot = joystick_policy.robot

    robot.limit_action_range = action_range # Set the action range for the robot
    if log_print:
        print("Action Space before applying bound", env.action_space)
    if frame_stack > 1:
        env = FrameStackWrapper(env, frame_stack, exclude_keys=["target_obs", "target_custom", "target_vel"])
    env = ClipActionToRange(env, robot.action_qpos_mins, robot.action_qpos_maxs) # Clip the action space to the range of the robot
    # env = gym.wrappers.ClipAction(env)

    if filter_actions > 0:
        env = ActionFilterWrapper(env, robot.control_timestep, robot, int(filter_actions))
    if action_history > 0:
        env = AddPreviousActions(env, action_history)
    
    if center_init_action:
        env = RescaleActionAsymmetric(env, -1.0, 1.0, robot.joint_qpos_init)
    else:
        env = gym.wrappers.RescaleAction(env, -1.0, 1.0)

    obs_space_before_flatten = env.observation_space
    if log_print:
        print("Obs Space before flattening", env.observation_space)
    env = gym.wrappers.FlattenObservation(env)
    env = wrap_gym(env, True)
    

    return env, obs_space_before_flatten

def get_formatted_register_env_entry_point(
    entry_point : Callable,
):
    def to_ret(
        make_env_kwargs: dict,
        make_env_kwargs_override: list[Callable[[dict], dict]],
        make_env_kwargs_callback:typing.Optional[Callable[[dict],dict]] = None,
    ) -> gym.Env:
        env_kargs = make_env_kwargs.copy()
        if make_env_kwargs_callback is not None:
            env_kargs = make_env_kwargs_callback(env_kargs)
        for override in make_env_kwargs_override:
            env_kargs = override(env_kargs)

        return entry_point(
            **env_kargs
        )
    return to_ret
    

def formatted_register_env(
    id_format:str,
    format_args:list,
    make_entry_point:Callable,
    register_kwargs_override:typing.Optional[dict],
    make_env_kargs: dict,
    make_env_kargs_override: list[Callable[[dict], dict]],
    make_env_kwargs_callback:typing.Optional[Callable[[dict],dict]] = None,
):
    call_kwargs = dict(
        id=id_format.format(*format_args),
        entry_point=get_formatted_register_env_entry_point(
            make_entry_point
        ),
        kwargs=dict(
            make_env_kwargs=make_env_kargs,
            make_env_kwargs_override=make_env_kargs_override.copy(),
            make_env_kwargs_callback=make_env_kwargs_callback
        )
    )
    if register_kwargs_override is not None:
        call_kwargs.update(register_kwargs_override)
    register(**call_kwargs)

def iter_formatted_register_env(
    id_format:str,
    make_entry_point:Callable,
    format_args_list:typing.List[typing.List[typing.Tuple[
        typing.Union[str, int, float],
        typing.Optional[dict],
        typing.Optional[Callable[[dict],dict]]
    ]]],
    base_register_kwargs:dict = {},
    base_make_env_kwargs:dict = {},
    base_make_env_kwargs_callback:typing.Optional[Callable[[dict],dict]] = None,
):
    def _iter_formatted_register(step, format_args : list[str | int | float] = [], register_kargs_override = {}, make_env_kargs_override : list[Callable[[dict],dict]] = []):
        if step >= len(format_args_list):
            register_kargs = base_register_kwargs.copy()
            register_kargs.update(register_kargs_override)
            formatted_register_env(id_format, format_args, make_entry_point, register_kargs, base_make_env_kwargs, make_env_kargs_override, base_make_env_kwargs_callback)
        else:
            for arg, register_kwargs_override, make_env_kwargs_override_fn in format_args_list[step]:
                
                format_args.append(arg)
                make_env_kargs_override.append(make_env_kwargs_override_fn)
                
                if register_kwargs_override is not None:
                    new_register_kwargs_override = register_kargs_override.copy()
                    new_register_kwargs_override.update(register_kwargs_override)
                else:
                    new_register_kwargs_override = register_kargs_override
                
                _iter_formatted_register(step+1, format_args, new_register_kwargs_override, make_env_kargs_override)
                format_args.pop()
                make_env_kargs_override.pop()
    _iter_formatted_register(0, [], {}, [])
