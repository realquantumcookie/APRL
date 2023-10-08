from ..robot import BaseWalker, BaseWalkerWithFootContact
from .joystick_interfaces import JoystickPolicyResetter, JoystickPolicyRewardProvider, JoystickPolicyTargetProvider, JoystickPolicyTerminationConditionProvider, JoystickPolicyTargetObservable
import numpy as np
from typing import Optional, Any
import transforms3d as tr3d

def normalize_rad(angle: float) -> float:
    # return np.arctan2(np.sin(angle),np.cos(angle))
    return ((angle + np.pi) % (2 * np.pi)) - np.pi

class JoystickPolicy:
    def __init__(
        self,
        robot: BaseWalker,
        reward_provider: JoystickPolicyRewardProvider,
        target_yaw_provider: JoystickPolicyTargetProvider,
        termination_providers: list[JoystickPolicyTerminationConditionProvider],
        truncation_providers: list[JoystickPolicyTerminationConditionProvider],
        resetters: list[JoystickPolicyResetter],
        initializers: list[JoystickPolicyResetter] = [],
        target_observable: Optional[JoystickPolicyTargetObservable] = None,
        enabled_observables : list[str] = [
            "joints_pos",
            "joints_vel",
            "imu",
            "sensors_local_velocimeter",
            "torques",
            "foot_contacts",
        ],
        lock_target: bool = False,
        enable_target_custom_obs = True
    ):
        self.robot = robot
        self.reward_provider = reward_provider
        self.target_yaw_provider = target_yaw_provider
        self.termination_providers = termination_providers
        self.truncation_providers = truncation_providers
        self.resetters = resetters
        self.initializers = initializers
        self.target_observable = target_observable
        self.enabled_observables = enabled_observables
        self.lock_target = lock_target
        self.enable_target_custom_obs = enable_target_custom_obs

        # Temporary Variables
        self._step_target_qpos = self.robot.get_joint_qpos()

        # Set up task-specific variables
        self._target_goal_world_delta = np.zeros(2)
        self._target_goal_local = np.zeros(2)
        self._target_yaw = 0.0
        self._target_delta_yaw = 0.0
        self._target_velocity = 0.0
        self._target_custom_data = None
        self._rew_step = 0.0
        self._info_dict = {}
        self._has_after_after_step = False
        self._termination_reason : Optional[JoystickPolicyTerminationConditionProvider] = None
        self._truncation_reason : Optional[JoystickPolicyTerminationConditionProvider] = None
        self._inited = False
    
    @property
    def has_after_after_step(self) -> bool:
        return self._has_after_after_step

    @property
    def control_timestep(self) -> float:
        return self.robot.control_timestep

    @control_timestep.setter
    def control_timestep(self, value: float) -> None:
        self.robot.control_timestep = value
    
    @property
    def last_info(self) -> dict[str,Any]:
        return self._info_dict.copy()

    @property
    def control_subtimestep(self) -> float:
        return self.robot.control_subtimestep
    
    @control_subtimestep.setter
    def control_subtimestep(self, value: float) -> None:
        self.robot.control_subtimestep = value

    @property
    def target_yaw(self) -> float:
        return self._target_yaw
    
    @property
    def target_delta_yaw(self) -> float:
        return self._target_delta_yaw

    @property
    def target_goal_world_delta(self) -> np.ndarray:
        return self._target_goal_world_delta.copy()
    
    @property
    def target_goal_local(self) -> np.ndarray:
        return self._target_goal_local.copy()
    
    @property
    def target_custom_data(self) -> Optional[Any]:
        return self._target_custom_data
    
    @property
    def target_goal_world_delta_unit(self) -> np.ndarray:
        norm_goal = np.linalg.norm(self._target_goal_world_delta)
        if norm_goal == 0.0:
            return np.zeros(2)
        else:
            return self._target_goal_world_delta / norm_goal
    
    @property
    def target_goal_local_unit(self) -> np.ndarray:
        norm_goal = np.linalg.norm(self._target_goal_local)
        if norm_goal == 0.0:
            return np.zeros(2)
        else:
            return self._target_goal_local / norm_goal
    

    def __update_target(self) -> float:
        new_target_goal_world_delta = self.target_yaw_provider.get_target_goal_world_delta(self.robot)[:2]
        new_target_velocity = self.target_yaw_provider.get_target_velocity(self.robot)
        _, _, yaw = self.robot.get_roll_pitch_yaw()
        inv_rotation_mat = np.array([
            [np.cos(yaw), np.sin(yaw)],
            [-np.sin(yaw), np.cos(yaw)]
        ])
        new_target_goal_local = inv_rotation_mat @ new_target_goal_world_delta

        new_target_yaw = np.arctan2(new_target_goal_world_delta[1], new_target_goal_world_delta[0]) if np.linalg.norm(new_target_goal_world_delta) > 0.0 else 0.0
        new_target_delta_yaw = normalize_rad(new_target_yaw - self.robot.get_roll_pitch_yaw()[2])
        change_in_abs_target_delta_yaw = self.__get_change_in_abs_target_delta_yaw()

        self._info_dict["target_yaw"] = new_target_yaw
        self._info_dict["target_delta_yaw"] = new_target_delta_yaw
        self._info_dict["target_goal_local_x"] = new_target_goal_local[0]
        self._info_dict["target_goal_local_y"] = new_target_goal_local[1]
        self._info_dict["target_goal_world_delta_x"] = new_target_goal_world_delta[0]
        self._info_dict["target_goal_world_delta_y"] = new_target_goal_world_delta[1]
        self._info_dict["change_in_abs_target_delta_yaw"] = change_in_abs_target_delta_yaw
        self._info_dict["abs_target_delta_yaw"] = np.abs(new_target_delta_yaw)
        self._info_dict["target_velocity"] = new_target_velocity

        self._target_yaw = new_target_yaw
        self._target_delta_yaw = new_target_delta_yaw
        self._target_goal_local = new_target_goal_local
        self._target_goal_world_delta = new_target_goal_world_delta
        self._target_custom_data = self.target_yaw_provider.get_target_custom_data()
        self._target_velocity = new_target_velocity
        return change_in_abs_target_delta_yaw
    
    def __get_change_in_abs_target_delta_yaw(self) -> float:
        new_target_delta_yaw = normalize_rad(self.target_yaw - self.robot.get_roll_pitch_yaw()[2])
        change_in_abs_target_delta_yaw = np.abs(new_target_delta_yaw) - np.abs(self._target_delta_yaw)
        return change_in_abs_target_delta_yaw
    
    def before_step(
        self,
        action: np.ndarray,
        random_state : np.random.RandomState
    ):
        self._step_target_qpos = action
        self.robot.apply_action(action)

    def get_reward(
        self
    ):
        return self._rew_step

    def after_step(
        self,
        random_state : np.random.RandomState
    ) -> dict[str,Any]:
        self._info_dict = {}
        self.robot.receive_observation()

        # Update the target yaw
        self.target_yaw_provider.step_target(
            self.robot,
            self._info_dict,
            random_state
        )
        self._has_after_after_step = self.target_yaw_provider.has_target_changed()
        if not self.lock_target and self._has_after_after_step:
            change_in_abs_target_delta_yaw = self.after_after_step(
                random_state
            )
        else:
            change_in_abs_target_delta_yaw = self.__update_target()

        # Gather info about velocity
        robot_v = self.robot.get_3d_linear_velocity()
        robot_v_norm = np.linalg.norm(robot_v)
        robot_v_to_goal = np.dot(
            robot_v[:2], self.target_goal_world_delta_unit
        )
        robot_v_local = self.robot.get_3d_local_velocity()
        robot_rpy = self.robot.get_roll_pitch_yaw()
        self._info_dict["velocity_norm"] = robot_v_norm
        self._info_dict["velocity_to_goal"] = robot_v_to_goal
        self._info_dict["velocity_local_x"] = robot_v_local[0]
        self._info_dict["velocity_local_y"] = robot_v_local[1]
        self._info_dict["velocity_local_z"] = robot_v_local[2]
        self._info_dict["roll"] = robot_rpy[0]
        self._info_dict["pitch"] = robot_rpy[1]
        self._info_dict["yaw"] = robot_rpy[2]
        self._info_dict["joint_torques"] = np.mean(np.abs(self.robot.get_joint_torques()))
        self._info_dict["joint_qvels"] = np.mean(np.abs(self.robot.get_joint_qvel()))
        self._info_dict["joint_qaccs"] = np.mean(np.abs(self.robot.get_joint_qacc()))
        self._info_dict["joint_velocities"] = np.mean(np.abs(self.robot.get_joint_qvel()))
        if hasattr(self.robot, "get_foot_force"):
            foot_force : np.ndarray = self.robot.get_foot_force()
            if foot_force.shape == (4,):
                foot_force_names = ["FR", "FL", "RR", "RL"]
            else:
                foot_force_names = list(range(foot_force.shape[0]))
            for i in range(len(foot_force_names)):
                self._info_dict["foot_force_" + foot_force_names[i]] = foot_force[i]

        self.reward_provider.step_reward(
            self.robot,
            self._step_target_qpos,
            self.target_goal_world_delta,
            self.target_goal_local,
            self.target_yaw,
            self.target_delta_yaw,
            self._target_velocity,
            robot_v_to_goal,
            change_in_abs_target_delta_yaw,
            self._target_custom_data,
            self.enable_target_custom_obs,
            self._info_dict,
            random_state
        )
        reward_perstep = self.reward_provider.get_reward()
        #assert reward_perstep is not None and reward_perstep != np.nan
        self._info_dict["reward_perstep"] = reward_perstep
        self._rew_step = reward_perstep

        # Step the target yaw observable
        if self.target_observable is not None:
            self.target_observable.step_target_obs(
                self.robot,
                self.target_goal_world_delta,
                self.target_goal_local,
                self.target_yaw,
                self.target_delta_yaw,
                self._target_velocity,
                robot_v_to_goal,
                change_in_abs_target_delta_yaw,
                self._target_custom_data,
                self.enable_target_custom_obs,
                self._info_dict,
                random_state
            )
        
        # Step resetters
        for resetter in self.resetters:
            resetter.step_resetter(
                self.robot,
                self.target_goal_world_delta,
                self.target_goal_local,
                self.target_yaw,
                self.target_delta_yaw,
                self._target_velocity,
                robot_v_to_goal,
                change_in_abs_target_delta_yaw,
                self._target_custom_data,
                self.enable_target_custom_obs,
                self._info_dict,
                random_state
            )

        # Step termination providers
        for termination_provider in self.termination_providers:
            termination_provider.step_termination_condition(
                self.robot,
                self.target_goal_world_delta,
                self.target_goal_local,
                self.target_yaw,
                self.target_delta_yaw,
                self._target_velocity,
                robot_v_to_goal,
                change_in_abs_target_delta_yaw,
                self._target_custom_data,
                self.enable_target_custom_obs,
                self._info_dict,
                random_state
            )
            if termination_provider.should_terminate():
                # print("Termination provider", termination_provider, "terminated the episode")
                self._termination_reason = termination_provider
                break
        
        # Step truncaiton providers
        for truncation_provider in self.truncation_providers:
            truncation_provider.step_termination_condition(
                self.robot,
                self.target_goal_world_delta,
                self.target_goal_local,
                self.target_yaw,
                self.target_delta_yaw,
                self._target_velocity,
                robot_v_to_goal,
                change_in_abs_target_delta_yaw,
                self._target_custom_data,
                self.enable_target_custom_obs,
                self._info_dict,
                random_state
            )
            if truncation_provider.should_terminate():
                # print("Truncation provider", truncation_provider, "truncated the episode")
                self._truncation_reason = truncation_provider
                break

        return self._info_dict.copy()

    def after_after_step(
        self,
        random_state : np.random.RandomState
    ):
        if self._has_after_after_step:
            self.target_yaw_provider.after_step_target(
                self.robot,
                self._info_dict,
                random_state
            )
            change_in_abs_target_delta_yaw = self.__update_target()
            robot_v = self.robot.get_3d_linear_velocity()
            robot_v_to_goal = np.dot(
                robot_v[:2], self.target_goal_world_delta_unit
            )
            # Step the target yaw observable
            if self.target_observable is not None:
                self.target_observable.step_target_obs(
                    self.robot,
                    self.target_goal_world_delta,
                    self.target_goal_local,
                    self.target_yaw,
                    self.target_delta_yaw,
                    self._target_velocity,
                    robot_v_to_goal,
                    change_in_abs_target_delta_yaw,
                    self._target_custom_data,
                    self.enable_target_custom_obs,
                    self._info_dict,
                    random_state
                )
            
            # self.reward_provider.step_ex(
            #     self.robot,
            #     self.target_goal_world_delta,
            #     self.target_goal_local,
            #     self.target_yaw,
            #     self.target_delta_yaw,
            #     robot_v_to_goal,
            #     change_in_abs_target_delta_yaw,
            #     self._target_custom_data,
            #     self.enable_target_custom_obs,
            #     self._info_dict,
            #     random_state
            # )
            # reward_perstep = self.reward_provider.get_reward()
            # #assert reward_perstep is not None and reward_perstep != np.nan
            # self._rew_step = reward_perstep

            self._has_after_after_step = False
            return change_in_abs_target_delta_yaw
        else:
            return 0.0
    
    def reset(self, random_state : np.random.RandomState) -> dict[str,Any]:
        self.robot.receive_observation()
        # Reset the info dict
        self._info_dict = {}

        # Reset the task-specific variables
        self._target_yaw = 0.0
        self._target_delta_yaw = 0.0
        self._has_after_after_step = False

        if not self._inited:
            self._inited = True
            for initializer in self.initializers:
                initializer.perform_reset(
                    self.robot,
                    self._info_dict,
                    self._termination_reason,
                    random_state
                )

        # call the resetters
        for resetter in self.resetters:
            resetter.perform_reset(
                self.robot,
                self._info_dict,
                self._termination_reason,
                random_state
            )

        # Reset the target yaw provider
        self.target_yaw_provider.reset_target(
            self.robot,
            self._info_dict,
            self._termination_reason,
            random_state
        )
        self.__update_target()

        # Reset target yaw obs
        if self.target_observable is not None:
            self.target_observable.reset_target_obs(
                self.robot,
                self.target_goal_world_delta,
                self.target_goal_local,
                self.target_yaw,
                self.target_delta_yaw,
                self._target_velocity,
                self._info_dict,
                self._target_custom_data,
                self.enable_target_custom_obs,
                self._termination_reason,
                random_state
            )

        # Reset reward provider
        self.reward_provider.reset_reward(
            self.robot,
            self._info_dict,
            self._termination_reason,
            random_state
        )

        # Reset termination providers
        for termination_provider in self.termination_providers:
            termination_provider.reset_termination_condition(
                self.robot,
                self._info_dict,
                self._termination_reason,
                random_state
            )
        
        # Reset truncation providers
        for truncation_provider in self.truncation_providers:
            truncation_provider.reset_termination_condition(
                self.robot,
                self._info_dict,
                self._termination_reason,
                random_state
            )
        
        self._termination_reason = None
        self._truncation_reason = None
        self._rew_step = 0.0

        # Reset the robot
        self.robot.reset()
        self.robot.receive_observation()

        for resetter in self.resetters:
            if hasattr(resetter, "last_position"):
                resetter.perform_reset(
                    self.robot,
                    self._info_dict,
                    self._termination_reason,
                    random_state
                )

        return self._info_dict.copy()
    
    def should_terminate(self) -> bool:
        return self._termination_reason is not None

    def should_truncate(self) -> bool:
        return self._truncation_reason is not None