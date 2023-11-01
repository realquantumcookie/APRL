import gym
from .register_helper import iter_formatted_register_env
from rail_walker_interface import JoystickPolicy, BaseWalker, WalkerVelocitySmoother, JoystickEnvImpl
from rail_real_walker import DummyVelocityEstimator, IntelRealsenseT265Estimator, IntelRealsenseT265EstimatorRemote, Go1ForwardKinematicsVelocityProvider, KalmanFilterFusedVelocityEstimator
from rail_real_walker.robots import Go1RealWalker
from typing import Any
from .wrappers import *
from ..joystick_policy import *
from ..joystick_policy_real import *
import numpy as np
from .register_config import *
import copy

def make_real_env(
    robot: BaseWalker,
    joystick_policy_parameters : dict[str,Any]
) -> gym.Env:
    print("Making real env")
    
    joystick_policy = JoystickPolicy(
        robot = robot,
        **joystick_policy_parameters
    )

    print("Joystick Policy Parameters", joystick_policy_parameters)
    
    env = JoystickEnvImpl(joystick_policy)
    return env

def make_sanity_lambda_real(
    kwargs: dict[str,Any]
) -> dict[str,Any]:
    kwargs['joystick_policy_parameters']['target_yaw_provider'] = JoystickPolicyForwardOnlyTargetProvider()
    return kwargs

def make_autojoystick_lambda_real(
    kwargs: dict[str,Any]
) -> dict[str,Any]:
    kwargs['joystick_policy_parameters']['target_yaw_provider'] = JoystickPolicyAutoJoystickTargetProvider()
    return kwargs

def make_realjoystick_lambda_real(
    kwargs: dict[str,Any]
) -> dict[str,Any]:
    kwargs['joystick_policy_parameters']['target_yaw_provider'] = JoystickPolicyGo1JoystickTargetProvider()
    kwargs['robot'] = WalkerVelocitySmoother(
        kwargs['robot'], 
        np.array([0.05,0.05,0.1,0.1,0.2,0.5])
    )
    return kwargs

def make_realjoystick_simple_lambda_real(
    kwargs: dict[str,Any]
) -> dict[str,Any]:
    kwargs['joystick_policy_parameters']['target_yaw_provider'] = JoystickPolicyGo1JoystickSimpleTargetProvider()
    kwargs['robot'] = WalkerVelocitySmoother(
        kwargs['robot'], 
        np.array([0.05,0.05,0.1,0.1,0.2,0.5])
    )
    return kwargs

def make_smoothed_target_lambda(
    kwargs: dict[str,Any]
) -> dict[str,Any]:
    kwargs['joystick_policy_parameters']['target_yaw_provider'] = JoystickPolicySmoothedTargetProvider(
        kwargs['joystick_policy_parameters']['target_yaw_provider'],
        max_rate=5.0/180.0*np.pi,
    )
    return kwargs

def make_limited_target_lambda(
    kwargs: dict[str,Any]
) -> dict[str,Any]:
    kwargs['joystick_policy_parameters']['target_yaw_provider'] = JoystickPolicyLimitedTargetProvider(
        kwargs['joystick_policy_parameters']['target_yaw_provider'],
        max_delta_angle=30.0/180.0*np.pi,
    )
    return kwargs

def make_target_delta_yaw_obs_lambda(
    kwargs: dict[str,Any]
) -> dict[str,Any]:
    kwargs['joystick_policy_parameters']['target_observable'] = JoystickPolicyTargetDeltaYawObservable()
    return kwargs

def make_cos_sin_target_delta_yaw_obs_lambda(
    kwargs: dict[str,Any]
) -> dict[str,Any]:
    kwargs['joystick_policy_parameters']['target_observable'] = JoystickPolicyCosSinTargetDeltaYawObservable()
    return kwargs

def make_target_delta_yaw_and_dist_obs_lambda(
    kwargs: dict[str,Any]
) -> dict[str,Any]:
    kwargs['joystick_policy_parameters']['target_observable'] = JoystickPolicyTargetDeltaYawAndDistanceObservable()
    return kwargs

def make_empty_obs_lambda(
    kwargs: dict[str,Any]
) -> dict[str,Any]:
    kwargs['joystick_policy_parameters']['target_observable'] = None
    return kwargs

def make_strict_reward_lambda(
    kwargs: dict[str,Any]
) -> dict[str, Any]:
    kwargs['joystick_policy_parameters']['reward_provider'] = JoystickPolicyStrictRewardProvider()
    return kwargs

def make_sep_reward_lambda(
    kwargs: dict[str,Any]
) -> dict[str, Any]:
    kwargs['joystick_policy_parameters']['reward_provider'] = JoystickPolicySeperateRewardProvider()
    return kwargs

iter_formatted_register_env(
    id_format = "{}{}Real-{}{}-{}-v0",
    make_entry_point = make_real_env,
    format_args_list = [
        [
            ("Go1", None, lambda kwargs: {
                **kwargs,
                "robot": Go1RealWalker(
                    velocity_estimator=IntelRealsenseT265Estimator(
                        x_axis_on_robot=np.array([-1, 0, 0]),
                        y_axis_on_robot=np.array([0, 0, 1]),
                        z_axis_on_robot=np.array([0, 1, 0]),
                    ),
                    power_protect_factor=CONFIG_REAL_POWER_PROTECT_FACTOR,
                    Kp=CONFIG_KP,
                    Kd=CONFIG_KD,
                    limit_action_range=CONFIG_ACTION_RANGE,
                    action_interpolation=CONFIG_ACTION_INTERPOLATION
                )
            })
        ],
        [
            ("Sanity", None, make_sanity_lambda_real),
            ("AutoJoystick", None, make_autojoystick_lambda_real),
            ("RealJoystick", None, make_realjoystick_lambda_real),
            ("RealJoystickSimple", None, make_realjoystick_simple_lambda_real)
        ],
        [
            ("Smoothed", None, make_smoothed_target_lambda),
            ("Limited", None, make_limited_target_lambda),
            ("", None, lambda kwargs: kwargs)
        ],
        [
            ("TDY", None, make_target_delta_yaw_obs_lambda),
            ("CosSinTDY", None, make_cos_sin_target_delta_yaw_obs_lambda),
            ("TDYDist", None, make_target_delta_yaw_and_dist_obs_lambda),
            ("Empty", None, make_empty_obs_lambda)
        ],
        [
            ("StrictRew", None, make_strict_reward_lambda),
            ("SepRew", None, make_sep_reward_lambda)
        ]
    ],
    base_register_kwargs = dict(
        
    ),
    base_make_env_kwargs = dict(
        
    ),
    base_make_env_kwargs_callback = lambda kwargs: {
        **kwargs,
        "joystick_policy_parameters": dict(
            termination_providers=[JoystickPolicyRollPitchTerminationConditionProvider(30/180*np.pi, -30/180*np.pi, 25/180 * np.pi)],
            truncation_providers=[],
            resetters=[JoystickPolicyManualResetter()],
            enabled_observables = [
                "joints_pos",
                "joints_vel",
                "imu",
                "sensors_local_velocimeter",
                "torques", 
                "foot_forces_normalized"
            ]
        )
    }
)

# For debugging real world reset policy

def lambda_sitting_real_env(
    kwargs
):
    new_kwargs = copy.copy(kwargs)
    new_kwargs["joystick_policy_parameters"]["resetters"] = [JoystickPolicyManualResetter(target_pose="sitting")]
    return new_kwargs

def lambda_standing_real_env(
    kwargs
):
    new_kwargs = copy.copy(kwargs)
    new_kwargs["joystick_policy_parameters"]["resetters"] = [JoystickPolicyManualResetter(target_pose="standing")]
    return new_kwargs

iter_formatted_register_env(
    id_format = "{}{}ResetReal-v0",
    make_entry_point = make_real_env,
    format_args_list = [
        [
            ("Go1", None, lambda kwargs: {
                **kwargs,
                "robot": Go1RealWalker(
                    velocity_estimator=DummyVelocityEstimator(),
                    power_protect_factor=CONFIG_REAL_POWER_PROTECT_FACTOR,
                    Kp=CONFIG_KP,
                    Kd=CONFIG_KD,
                    limit_action_range=CONFIG_ACTION_RANGE,
                    action_interpolation=CONFIG_ACTION_INTERPOLATION
                )
            }),
        ],
        [
            ("Sitting", None, lambda_sitting_real_env),
            ("Standing", None, lambda_standing_real_env)
        ]
    ],
    base_register_kwargs = dict(
        
    ),
    base_make_env_kwargs = dict(
        
    ),
    base_make_env_kwargs_callback = lambda kwargs: {
        **kwargs,
        "joystick_policy_parameters": dict(
            reward_provider=JoystickPolicyStrictRewardProvider(),
            target_yaw_provider=JoystickPolicyForwardOnlyTargetProvider(),
            termination_providers=[],
            truncation_providers=[],
            resetters=[],
            target_observable = None,
            enabled_observables = ["joints_pos", "imu"]
        )
    }
)