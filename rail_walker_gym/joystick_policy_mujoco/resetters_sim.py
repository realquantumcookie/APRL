import numpy as np
from rail_walker_interface import JoystickPolicyResetter, JoystickPolicyTerminationConditionProvider, BaseWalkerInSim, BaseWalker
from rail_mujoco_walker import RailSimWalkerDMControl
import transforms3d as tr3d
from typing import Any, Optional
import copy

class JoystickPolicyPointInSimResetter(JoystickPolicyResetter[BaseWalkerInSim]):
    def __init__(
        self,
        respawn_pos: np.ndarray = np.zeros(2),
        respawn_yaw: float = 0.0,
    ):
        super().__init__()
        self.respawn_pos = respawn_pos
        self.respawn_yaw = respawn_yaw
        self._last_should_crouch = False

    def get_respawn_pose(
        self,
        Robot: BaseWalkerInSim,
        randomState: np.random.RandomState
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.respawn_pos, np.array([0, 0, self.respawn_yaw])
    
    def step_resetter(
        self, 
        Robot: BaseWalkerInSim, 
        target_goal_world_delta: np.ndarray, 
        target_goal_local: np.ndarray, 
        target_yaw: float, 
        target_delta_yaw: float, 
        target_velocity: float,
        velocity_to_goal: float, 
        change_in_abs_target_delta_yaw: float, 
        target_custom_data: Optional[Any], 
        enable_target_custom_obs : bool,
        info_dict: dict[str, Any], 
        randomState: np.random.RandomState
    ) -> None:
        if target_custom_data is not None and "should_crouch" in target_custom_data and target_custom_data["should_crouch"]:
            self._last_should_crouch = True
        else:
            self._last_should_crouch = False
    
    def perform_reset(
        self, 
        Robot: BaseWalkerInSim, 
        info_dict: dict[str,Any], 
        termination_provider_triggered: JoystickPolicyTerminationConditionProvider,
        randomState: np.random.RandomState
    ) -> None:
        respawn_pos, respawn_roll_pitch_yaw = self.get_respawn_pose(Robot, randomState)
        target_qpos = Robot.joint_qpos_init if not self._last_should_crouch else Robot.joint_qpos_crouch
        
        if respawn_pos is not None and respawn_roll_pitch_yaw is not None:
            respawn_quat = tr3d.euler.euler2quat(*respawn_roll_pitch_yaw)
            Robot.reset_2d_location(respawn_pos, respawn_quat, target_qpos)


class JoystickPolicyLastPositionAndYawResetter(JoystickPolicyPointInSimResetter):
    def __init__(
        self,
        random_respawn_yaw : bool = False,
        respawn_yaw_override : Optional[float] = None,
        position_if_init : Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.random_respawn_yaw = random_respawn_yaw
        self.respawn_yaw_override = respawn_yaw_override
        self.last_position = position_if_init
        self.last_yaw = respawn_yaw_override
    
    def get_respawn_pose(self, Robot: BaseWalkerInSim, randomState: np.random.RandomState) -> tuple[np.ndarray, np.ndarray]:
        if self.last_yaw is None:
            respawn_yaw = randomState.uniform(-np.pi, np.pi)
        else:
            respawn_yaw = self.last_yaw
        
        return self.last_position, np.array([0, 0, respawn_yaw])
    
    def step_resetter(
        self, 
        Robot: BaseWalkerInSim, 
        target_goal_world_delta: np.ndarray, 
        target_goal_local: np.ndarray, 
        target_yaw: float, 
        target_delta_yaw: float, 
        target_velocity: float,
        velocity_to_goal: float, 
        change_in_abs_target_delta_yaw: float, 
        target_custom_data: Optional[Any], 
        enable_target_custom_obs : bool,
        info_dict: dict[str, Any], 
        randomState: np.random.RandomState
    ) -> None:
        if self.respawn_yaw_override is not None:
            self.last_yaw = self.respawn_yaw_override
        elif self.random_respawn_yaw:
            self.last_yaw = None
        else:
            self.last_yaw = Robot.get_roll_pitch_yaw()[2]

        self.last_position = Robot.get_3d_location()[:2]
        super().step_resetter(Robot, target_goal_world_delta, target_goal_local, target_yaw, target_delta_yaw, target_velocity, velocity_to_goal, change_in_abs_target_delta_yaw, target_custom_data, enable_target_custom_obs, info_dict, randomState)

class ResetPolicyResetter(JoystickPolicyResetter[RailSimWalkerDMControl]):
    def __init__(
        self,
        respawn_pos: np.ndarray = np.zeros(2),
        init_dist: np.ndarray = np.array([0.2, 0.2, 0.6]),
    ):
        self.respawn_pos = respawn_pos
        self._init_dist = init_dist
        super().__init__()
    
    def perform_reset(
        self, 
        Robot: RailSimWalkerDMControl, 
        info_dict: dict[str,Any], 
        termination_provider_triggered: JoystickPolicyTerminationConditionProvider,
        randomState: np.random.RandomState
    ) -> None:
        initialization = randomState.choice(['sitting', 'standing', 'fallen'], p=list(self._init_dist))
        if initialization == 'sitting':
            target_pose = np.asarray([0.0 / 180 * np.pi, 70.0/180*np.pi, -150.0 / 180 * np.pi] * 4)
            target_location = np.asarray([0, 0, 0.02])
            target_quat = None
        elif initialization == 'standing':
            target_pose = Robot.joint_qpos_init
            target_location = np.array([0, 0, 0.21])
            target_quat = None
        elif initialization == 'fallen':
            nominal_pose = copy.deepcopy(Robot.joint_qpos_init)
            target_location = np.array([0, 0, randomState.uniform(low=0.4, high=0.5)])
            root_rot = randomState.uniform(low=[-3 * np.pi / 4, -3 * np.pi / 4, -np.pi],
                                            high=[3 * np.pi / 4, 3 * np.pi / 4, np.pi])
            target_quat = tr3d.euler.euler2quat(root_rot[0], root_rot[1], root_rot[2])

            joint_lim_low = np.asarray([-0.802851455917, -1.0471975512, -2.69653369433] * 4)
            joint_lim_high = np.asarray([0.802851455917, 4.18879020479, -0.916297857297] * 4)
            joint_pose_size = len(joint_lim_low)

            joint_dir = randomState.randint(0, 2, joint_pose_size).astype(np.float32)
            lim_pose = (1.0 - joint_dir) * joint_lim_low + joint_dir * joint_lim_high

            pose_lerp = randomState.uniform(low=0, high=1, size=joint_pose_size)
            pose_lerp = pose_lerp * pose_lerp * pose_lerp
            target_pose = (1.0 - pose_lerp) * nominal_pose + pose_lerp * lim_pose
            

        Robot.reset_dropped(
            target_pose,
            target_location,
            target_quat,
            settle = initialization == 'fallen'
        )
