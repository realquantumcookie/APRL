import gym
from rail_mujoco_walker.robots.sim_robot import RailSimWalkerDMControl
from rail_walker_interface import BaseWalker, JoystickPolicy, JoystickEnvironment, JoystickPolicyTargetProvider, JoystickPolicyTerminationConditionProvider
from rail_mujoco_walker import DMWalkerForRailSimWalker, RailSimWalkerDMControl
from typing import List, Optional, Tuple, Dict, Any
import numpy as np

class JoystickPolicyRollout:
    def __init__(self) -> None:
        self.joint_qpos : List[np.ndarray] = []
        self.joint_qvel : List[np.ndarray] = []
        self.joint_torques : List[np.ndarray] = []
        self.framequats : List[np.ndarray] = []
        self.linear_3d_velocities : List[np.ndarray] = []
        self.angular_3d_velocities : List[np.ndarray] = []
        self.target_world_deltas : List[np.ndarray] = []
        self.target_linear_velocities : List[float] = []
        self.terminations : List[bool] = []
        self.truncations : List[bool] = []

    def append(
        self,
        joystick_policy: JoystickPolicy
    ):
        self.joint_qpos.append(joystick_policy.robot.get_joint_qpos())
        self.joint_qvel.append(joystick_policy.robot.get_joint_qvel())
        self.joint_torques.append(joystick_policy.robot.get_joint_torques())
        self.framequats.append(joystick_policy.robot.get_framequat_wijk())
        self.linear_3d_velocities.append(joystick_policy.robot.get_3d_linear_velocity())
        self.angular_3d_velocities.append(joystick_policy.robot.get_3d_angular_velocity())
        self.target_world_deltas.append(joystick_policy.target_goal_world_delta())
        self.target_linear_velocities.append(joystick_policy._target_velocity)
        self.terminations.append(joystick_policy.should_terminate())
        self.truncations.append(joystick_policy.should_truncate())
    
    @property
    def dones(self) -> List[bool]:
        return [truncation or termination for truncation, termination in zip(self.truncations, self.terminations)]

    def current_episode(self):
        dones = self.dones
        last_done = len(self.dones) - 1
        last_start = last_done
        while last_start > 0 and not dones[last_start - 1]:
            last_start -= 1
        
        ret = __class__()
        ret.joint_qpos = self.joint_qpos[last_start:last_done+1]
        ret.joint_qvel = self.joint_qvel[last_start:last_done+1]
        ret.joint_torques = self.joint_torques[last_start:last_done+1]
        ret.framequats = self.framequats[last_start:last_done+1]
        ret.linear_3d_velocities = self.linear_3d_velocities[last_start:last_done+1]
        ret.angular_3d_velocities = self.angular_3d_velocities[last_start:last_done+1]
        ret.target_world_deltas = self.target_world_deltas[last_start:last_done+1]
        ret.target_linear_velocities = self.target_linear_velocities[last_start:last_done+1]
        ret.terminations = self.terminations[last_start:last_done+1]
        ret.truncations = self.truncations[last_start:last_done+1]
        return ret

    def last_episode(self):
        # Find the last done
        last_done = len(self.dones) - 1
        while last_done > 0 and not self.dones[last_done]:
            last_done -= 1

        last_start = last_done
        while last_start > 0 and not self.dones[last_start - 1]:
            last_start -= 1
        
        # Return the last episode
        ret = __class__()
        ret.joint_qpos = self.joint_qpos[last_start:last_done+1]
        ret.joint_qvel = self.joint_qvel[last_start:last_done+1]
        ret.joint_torques = self.joint_torques[last_start:last_done+1]
        ret.framequats = self.framequats[last_start:last_done+1]
        ret.linear_3d_velocities = self.linear_3d_velocities[last_start:last_done+1]
        ret.angular_3d_velocities = self.angular_3d_velocities[last_start:last_done+1]
        ret.target_world_deltas = self.target_world_deltas[last_start:last_done+1]
        ret.target_linear_velocities = self.target_linear_velocities[last_start:last_done+1]
        ret.terminations = self.terminations[last_start:last_done+1]
        ret.truncations = self.truncations[last_start:last_done+1]
        return ret
    
    def export_npz(self, path : str) -> None:
        np.savez_compressed(
            path, 
            joint_qpos=self.joint_qpos,
            joint_qvel=self.joint_qvel,
            joint_torques=self.joint_torques,
            framequats=self.framequats,
            linear_3d_velocities=self.linear_3d_velocities,
            angular_3d_velocities=self.angular_3d_velocities,
            target_world_deltas=self.target_world_deltas,
            target_linear_velocities=self.target_linear_velocities,
            terminations=self.terminations,
            truncations=self.truncations
        )
    
    def is_current_episode_empty(self) -> bool:
        dones = self.dones
        return len(dones) == 0 or dones[-1]

    def clear(self) -> None:
        self.joint_qpos.clear()
        self.joint_qvel.clear()
        self.joint_torques.clear()
        self.framequats.clear()
        self.linear_3d_velocities.clear()
        self.angular_3d_velocities.clear()
        self.target_world_deltas.clear()
        self.target_linear_velocities.clear()
        self.terminations.clear()
        self.truncations.clear()

    @staticmethod
    def import_npz(path : str):
        roll = __class__()
        data = np.load(path)
        roll.joint_qpos = list(data["joint_qpos"])
        roll.joint_qvel = list(data["joint_qvel"])
        roll.joint_torques = list(data["joint_torques"])
        roll.framequats = list(data["framequats"])
        roll.linear_3d_velocities = list(data["linear_3d_velocities"])
        roll.angular_3d_velocities = list(data["angular_3d_velocities"])
        roll.target_world_deltas = list(data["target_world_deltas"])
        roll.target_linear_velocities = list(data["target_linear_velocities"])
        roll.terminations = list(data["terminations"])
        roll.truncations = list(data["truncations"])
        return roll
    
    def __len__(self) -> int:
        return len(self.dones)

class JoystickPolicyRecorderWrapper(gym.Wrapper, JoystickEnvironment):
    def __init__(self, env: gym.Env):
        assert hasattr(env, "joystick_policy"), "env must have a joystick_policy attribute"
        assert isinstance(env.joystick_policy, JoystickPolicy), "env.joystick_policy must be a JoystickPolicy"
        gym.Wrapper.__init__(self, env)
        JoystickEnvironment.__init__(self)
        self.recorded_joystick_policy_rollout = JoystickPolicyRollout()

    @property
    def joystick_policy(self) -> JoystickPolicy:
        return self.env.joystick_policy
    
    def set_joystick_policy(self, joystick_policy: JoystickPolicy):
        self.env.set_joystick_policy(joystick_policy)

    def step(self, action: Any) -> Tuple[Any, float, bool, dict]:
        ret = self.env.step(action)
        self.recorded_joystick_policy_rollout.append(
            self.joystick_policy
        )
        return ret
    
    def reset(self, *args, **kwargs) -> Any:
        ret = self.env.reset(*args, **kwargs)
        self.recorded_joystick_policy_rollout.append(
            self.joystick_policy
        )
        return ret
    
class JoystickPolicyReplayTargetProvider(JoystickPolicyTargetProvider[RailSimWalkerDMControl], JoystickPolicyTerminationConditionProvider[RailSimWalkerDMControl]):
    def __init__(
        self,
        joystick_policy_rollout: JoystickPolicyRollout
    ):
        JoystickPolicyTargetProvider.__init__(self)
        JoystickPolicyTerminationConditionProvider.__init__(self)

        self.joystick_policy_rollout = joystick_policy_rollout

        self._is_target_velocity_fixed = True
        for target_velocity in self.joystick_policy_rollout.target_linear_velocities:
            if target_velocity != self.joystick_policy_rollout.target_linear_velocities[0]:
                self._is_target_velocity_fixed = False
                break
        
        self._current_step = 0
    
    def get_target_goal_world_delta(self, Robot: RailSimWalkerDMControl) -> np.ndarray:
        return self.joystick_policy_rollout.target_world_deltas[self._current_step % len(self.joystick_policy_rollout)]

    def get_target_velocity(self, Robot: RailSimWalkerDMControl) -> float:
        return self.joystick_policy_rollout.target_linear_velocities[self._current_step % len(self.joystick_policy_rollout)]
    
    def is_target_velocity_fixed(self) -> bool:
        return self._is_target_velocity_fixed

    def step_target(
        self, 
        Robot: RailSimWalkerDMControl, 
        info_dict: dict[str, Any], 
        randomState: np.random.RandomState
    ) -> None:
        self._current_step += 1
        current_position = Robot.get_3d_location()
        recorded_velocity = self.joystick_policy_rollout.linear_3d_velocities[self._current_step % len(self.joystick_policy_rollout)]
        new_2d_location = current_position[:2] + recorded_velocity[:2] * Robot.control_timestep
        Robot.reset_2d_location_with_qpos(
            new_2d_location,
            self.joystick_policy_rollout.framequats[self._current_step % len(self.joystick_policy_rollout)],
            self.joystick_policy_rollout.joint_qpos[self._current_step % len(self.joystick_policy_rollout)]
        )


    def reset_target(
        self, 
        Robot: RailSimWalkerDMControl, 
        info_dict: dict[str, Any], 
        termination_provider_triggered: JoystickPolicyTerminationConditionProvider, 
        randomState: np.random.RandomState
    ) -> None:
        self._current_step += 1 if self._current_step == 0 else 0
        Robot.reset_2d_location_with_qpos(
            np.zeros(2),
            self.joystick_policy_rollout.framequats[self._current_step % len(self.joystick_policy_rollout)],
            self.joystick_policy_rollout.joint_qpos[self._current_step % len(self.joystick_policy_rollout)]
        )

    def reset_termination_condition(
        self, 
        Robot: RailSimWalkerDMControl, 
        info_dict: dict[str, Any], 
        termination_provider_triggered, 
        randomState: np.random.RandomState
    ) -> None:
        return

    def step_termination_condition(
        self, 
        Robot: RailSimWalkerDMControl, 
        target_goal_world_delta: np.ndarray, 
        target_goal_local: np.ndarray, 
        target_yaw: float, 
        target_delta_yaw: float, 
        target_velocity: float, 
        velocity_to_goal: float, 
        change_in_abs_target_delta_yaw: float, 
        target_custom_data: Optional[Any], 
        enable_target_custom_obs: bool, 
        info_dict: dict[str, Any], 
        randomState: np.random.RandomState
    ) -> None:
        return
    
    def should_terminate(self) -> bool:
        return self.joystick_policy_rollout.dones[self._current_step % len(self.joystick_policy_rollout)]
