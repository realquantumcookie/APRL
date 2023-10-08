from functools import cached_property
import typing

import numpy as np
from dm_control import composer, mjcf, mujoco
from dm_control.composer.observation import observable
from dm_control.locomotion.walkers import base
from dm_control.utils.transformations import quat_to_euler
from dm_env import specs
from rail_walker_interface import BaseWalkerInSim, BaseWalker, BaseWalkerWithFootContact
from dm_control.mjcf import Physics, Element
import transforms3d as tr3d
import os
from ..utils import find_dm_control_non_contacting_height, settle_physics
from collections import deque

SIM_ASSET_DIR = os.path.join(os.path.dirname(__file__), '../resources')

"""
BaseRobotObservables is a class that defines the observables of the robot.
Note that the observables defined in the inherited class will be included in the observables of the robot.
"""
class DMWalkerForRailSimWalkerObservables(composer.Observables): #(base.WalkerObservables):
    def __init__(self, entity : "DMWalkerForRailSimWalker", foot_contact_threshold : np.ndarray, delayed_substep : int = 20):
        self._delayed_substep = delayed_substep
        super().__init__(entity)
        self._entity = entity
        self._foot_contact_threshold = foot_contact_threshold
    
    @composer.observable
    def joints_pos(self):
        #return observable.MJCFFeature('qpos', self._entity.observable_joints)
        return self._entity.get_delayed_observable(
            "joints_pos",
            delay_substeps=self._delayed_substep,
            lambda_fn=lambda physics: physics.bind(self._entity.observable_joints).qpos
        )

    @composer.observable
    def joints_vel(self):
        #return observable.MJCFFeature('qvel', self._entity.observable_joints)
        return self._entity.get_delayed_observable(
            "joints_vel",
            delay_substeps=self._delayed_substep,
            lambda_fn=lambda physics: physics.bind(self._entity.observable_joints).qvel
        )

    @composer.observable
    def sensors_gyro(self):
        #return observable.MJCFFeature('sensordata',self._entity.mjcf_model.sensor.gyro)
        return self._entity.get_delayed_observable(
            "sensors_gyro",
            delay_substeps=self._delayed_substep,
            lambda_fn=lambda physics: physics.bind(self._entity.mjcf_model.sensor.gyro).sensordata
        )

    @composer.observable
    def sensors_framequat(self):
        # return observable.MJCFFeature('sensordata',self._entity.mjcf_model.sensor.framequat)
        return self._entity.get_delayed_observable(
            "sensors_framequat",
            delay_substeps=self._delayed_substep,
            lambda_fn=lambda physics: physics.bind(self._entity.mjcf_model.sensor.framequat).sensordata
        )

    @composer.observable 
    def torques(self):
        # return observable.MJCFFeature('force', self._entity.actuators)
        return self._entity.get_delayed_observable(
            "torques",
            delay_substeps=self._delayed_substep,
            lambda_fn=lambda physics: physics.bind(self._entity.actuators).force
        )
    
    @composer.observable
    def foot_forces(self):
        #return observable.MJCFFeature('sensordata', self._entity.mjcf_model.sensor.touch)
        return self._entity.get_delayed_observable(
            "foot_forces",
            delay_substeps=self._delayed_substep,
            lambda_fn=lambda physics: physics.bind(self._entity.mjcf_model.sensor.touch).sensordata
        )

    @composer.observable
    def foot_forces_normalized(self):
        # return observable.Generic(
        #     lambda physics: (physics.bind(self._entity.mjcf_model.sensor.touch).sensordata / foot_force_max_values).astype(np.float32)
        # )
        return self._entity.get_delayed_observable(
            "foot_forces_normalized",
            delay_substeps=self._delayed_substep,
            lambda_fn=lambda physics: (physics.bind(self._entity.mjcf_model.sensor.touch).sensordata / 50.0).astype(np.float32)
        )

    @composer.observable
    def foot_forces_normalized_masked(self):
        foot_force_max_values = np.array([100.0, 100.0, 60.0, 60.0])
        def lambda_func(physics):
                foot_forces = physics.bind(self._entity.mjcf_model.sensor.touch).sensordata
                normalized = np.array(foot_forces / foot_force_max_values, dtype=np.float32)
                normalized[-1] = 0.0
                return normalized
        # return observable.Generic(lambda_func)
        return self._entity.get_delayed_observable(
            "foot_forces_normalized_masked",
            delay_substeps=self._delayed_substep,
            lambda_fn=lambda_func
        )

    @composer.observable
    def foot_contacts(self):
        # return observable.Generic(
        #     lambda physics: (physics.bind(self._entity.mjcf_model.sensor.touch).sensordata >= self._foot_contact_threshold).astype(np.float32)
        # )
        return self._entity.get_delayed_observable(
            "foot_contacts",
            delay_substeps=self._delayed_substep,
            lambda_fn=lambda physics: (physics.bind(self._entity.mjcf_model.sensor.touch).sensordata >= self._foot_contact_threshold).astype(np.float32)
        )

    @composer.observable
    def sensors_local_velocimeter(self):
        # return observable.Generic(
        #     lambda physics: self._entity.get_velocity(physics)
        # )
        return self._entity.get_delayed_observable(
            "sensors_local_velocimeter",
            delay_substeps=self._delayed_substep,
            lambda_fn=lambda physics: self._entity.get_velocity(physics)
        )

    @composer.observable
    def sensors_local_velocimeter_x(self):
        return self._entity.get_delayed_observable(
            "sensors_local_velocimeter_x",
            delay_substeps=self._delayed_substep,
            lambda_fn=lambda physics: self._entity.get_velocity(physics)[0:1]
        )

    @composer.observable
    def imu(self):
        # return observable.Generic(lambda physics: self._entity.get_imu(physics))
        return self._entity.get_delayed_observable(
            "imu",
            delay_substeps=self._delayed_substep,
            lambda_fn=lambda physics: self._entity.get_imu(physics)
        )
    
    @composer.observable
    def sensors_accelerometer(self):
        # return observable.MJCFFeature('sensordata', self._entity.mjcf_model.sensor.accelerometer)
        return self._entity.get_delayed_observable(
            "sensors_accelerometer",
            delay_substeps=self._delayed_substep,
            lambda_fn=lambda physics: physics.bind(self._entity.mjcf_model.sensor.accelerometer).sensordata
        )

    # @composer.observable
    # def body_height(self):
    #     return observable.MJCFFeature('xpos', self._entity.root_body)[2]
    
    # @composer.observable
    # def body_position(self):
    #     return observable.MJCFFeature('xpos', self._entity.root_body)

    # Semantic groupings of Walker observables.
    def _collect_from_attachments(self, attribute_name):
        out = []
        for entity in self._entity.iter_entities(exclude_self=True):
            out.extend(getattr(entity.observables, attribute_name, []))
        return out

    @property
    def kinematic_sensors(self):
        return (
            [
                self.sensors_gyro,
                self.sensors_accelerometer,
                self.sensors_local_velocimeter,
                self.sensors_framequat
            ] +
            self._collect_from_attachments('kinematic_sensors')
        )

    @property
    def dynamic_sensors(self):
        return self._collect_from_attachments('dynamic_sensors')

    @property
    def proprioception(self):
        return ([self.joints_pos, self.joints_vel, self.torques] +
                self._collect_from_attachments('proprioception'))


class DMWalkerForRailSimWalker(base.Walker):
    # For some reason mujoco failed to read the <default/motor> flag in XML file, hard-coding this into the code.
    DEFAULT_CTRL_LIMIT_MIN = -33.5
    DEFAULT_CTRL_LIMIT_MAX = 33.5

    def __init__(self, XML_FILE : str, action_interpolation : bool, power_protect_factor : float = 0.1, *args, **kwargs):
        assert power_protect_factor > 0.0 and power_protect_factor <= 1.0
        self._XML_FILE = XML_FILE
        self.target_action : typing.Optional[np.ndarray] = None
        self.prev_action : typing.Optional[np.ndarray] = None
        self.last_obs : typing.Dict[str,typing.Any] = None
        self._control_timestep = 0.05
        self._control_subtimestep = 0.002
        self._substep_count = 0
        self.action_interpolation = action_interpolation
        self.power_protect_factor = power_protect_factor
        self._last_physics : typing.Optional[Physics] = None
        super().__init__(*args,**kwargs)

    def _build(
            self,
            name: typing.Optional[str] = "robot",
            kp: float = 60,
            kd: float = 6,
            foot_contact_threshold = np.array([10,10,10,10]),
            delay_substeps : int = 10,
        ):
        self._mjcf_root = mjcf.from_path(self._XML_FILE)
        if name:
            self._mjcf_root.model = name
        # Find MJCF elements that will be exposed as attributes.
        self._root_body = self._mjcf_root.find('body', 'trunk')
        self._root_body.pos[-1] = 0.125

        self._foot_sites = [
            self._mjcf_root.find('site', f'{joint_name}_foot_site')
            for joint_name in ["FR", "FL", "RR", "RL"]
        ]

        self._joints : list[Element] = [] #self.mjc_model.find_all('joint')

        self._actuators : list[Element] = self.mjcf_model.find_all('actuator')

        for actuator in self._actuators:
            self._joints.append(actuator.joint)

        self.feet_geom_names = ["RL", "RR", "FL", "FR"]
        # self.feet_geoms = [self.root_body.find("geom", name) for name in feet_geom_names]
        self._foot_friction = 0.8

        self.kp = kp
        self.kd = kd

        self.last_obs = {
            "3d_location": np.zeros(3),
            "3d_linear_velocity": np.zeros(3),
            "3d_local_velocity": np.zeros(3),
            "angular_velocity": np.zeros(3),
            "framequat_wijk": tr3d.euler.euler2quat(0,0,0),
            "roll_pitch_yaw": np.zeros(3),
            "3d_acceleration_local": np.zeros(3),
            "joint_pos": np.zeros(len(self._joints)),
            "joint_vel": np.zeros(len(self._joints)),
            "joint_acc" : np.zeros(len(self._joints)),
            "foot_force": np.zeros(4),
            "torques": np.zeros(len(self._actuators)),
        }
        self._delayed_observable_lambdas : typing.Dict[str, typing.Callable[[Physics], typing.Any]] = {}
        self._delayed_observable_queues : typing.Dict[str, deque] = {}
        self._last_step_position = None
        self._built_observable = DMWalkerForRailSimWalkerObservables(self, foot_contact_threshold, delay_substeps)
        self._delay_substeps = delay_substeps
        delayed_joint_acc_observable = self.get_delayed_observable(
            'joint_acc',
            delay_substeps,
            lambda physics : physics.bind(self.observable_joints).qacc
        )

    def get_delayed_observable(self, name : str, delay_substeps : int, lambda_fn : typing.Callable[[Physics], typing.Any]) -> observable.Generic:
        self._delayed_observable_lambdas[name] = lambda_fn
        self._delayed_observable_queues[name] = deque(maxlen=max(delay_substeps,1))
        def get_obs(physics : Physics):
            if len(self._delayed_observable_queues[name]) == 0:
                return self._delayed_observable_lambdas[name](physics)
            else:
                return self._delayed_observable_queues[name][0]
        return observable.Generic(get_obs)
    
    def read_delayed_observables(self, name: str):
        if name not in self._delayed_observable_queues:
            raise ValueError(f"Observable {name} not found.")
        if len(self._delayed_observable_queues[name]) == 0:
            return self._delayed_observable_lambdas[name](self._last_physics)
        else:
            return self._delayed_observable_queues[name][0]

    def action_spec(self, physics : Physics):
        minimum = []
        maximum = []
        for actuator in self.actuators:
            joint = actuator.joint

            joint_range = physics.bind(joint).range
            minimum.append(joint_range[0])
            maximum.append(joint_range[1])

        return specs.BoundedArray(
            shape=(len(minimum),),
            dtype=np.float32,
            minimum=minimum,
            maximum=maximum
            #name='\t'.join([actuator.name for actuator in self.actuators])
        )

    @cached_property
    def ctrllimits(self):
        minimum = []
        maximum = []
        for actuator in self.actuators:
            if actuator.ctrllimited == "true" or actuator.ctrllimited:
                minimum.append(actuator.ctrlrange[0])
                maximum.append(actuator.ctrlrange[1])
            elif actuator.ctrllimited is None:
                minimum.append(self.DEFAULT_CTRL_LIMIT_MIN)
                maximum.append(self.DEFAULT_CTRL_LIMIT_MAX)
            else:
                # no limit
                minimum.append(-np.inf)
                maximum.append(np.inf)

        return minimum, maximum

    def apply_action(self, physics : Physics, desired_qpos : np.ndarray, random_state : np.random.RandomState):
        joints_bind = physics.bind(self.joints)
        qpos = joints_bind.qpos
        qvel = joints_bind.qvel

        action = self.kp * (desired_qpos - qpos) - self.kd * qvel
        minimum, maximum = self.ctrllimits
        minimum = np.asarray(minimum)
        maximum = np.asarray(maximum)

        minimum *= self.power_protect_factor
        maximum *= self.power_protect_factor
        action = np.clip(action, minimum, maximum)
        physics.bind(self.actuators).ctrl = action

    def set_target_action(self, target_action : np.ndarray):
        # print(f"New action {target_action} set, last substep count: ", self._substep_count)
        self.prev_action = self.target_action
        self.target_action = target_action
        self._substep_count = 0

    def _build_observables(self):
        return self._built_observable
        
    def get_imu(self, physics):
        quat = physics.bind(self.mjcf_model.sensor.framequat).sensordata
        roll, pitch, yaw = quat_to_euler(quat)

        gyro = physics.bind(self.mjcf_model.sensor.gyro).sensordata
        dr, dp, dy = gyro

        return np.array([roll, pitch, dr, dp])

    @property
    def root_body(self):
        return self._root_body

    @property
    def foot_friction(self):
        friction_vals = []
        for geom_name in self.feet_geom_names:
            friction_vals.append(self._last_physics.named.model.geom_friction[f"robot/{geom_name}"][0])
        return np.mean(friction_vals)
        # return self._foot_friction
    
    @foot_friction.setter
    def foot_friction(self, value : float):
        print("here")
        for geom_name in self.feet_geom_names:
            self._last_physics.model.geom('robot/'+geom_name).friction[0] = value
        # for geom in self.feet_geoms:
        #     geom.friction = (value, 0.02, 0.01)
        self._foot_friction = value

    @property
    def joints(self):
        """List of joint elements belonging to the arm."""
        return self._joints

    @property
    def observable_joints(self):
        return self._joints

    @property
    def actuators(self):
        """List of actuator elements belonging to the arm."""
        return self._actuators

    @property
    def mjcf_model(self):
        """Returns the `mjcf.RootElement` object corresponding to this robot."""
        return self._mjcf_root

    @composer.cached_property
    def root_body_linvel_sensor(self):
        return self._mjcf_root.find("sensor", "trunk_linvel")

    def get_roll_pitch_yaw_rad(self, physics : Physics):
        quat = self.get_framequat(physics) #physics.bind(self.mjcf_model.sensor.framequat).sensordata
        return quat_to_euler(quat)

    def get_velocity(self, physics : Physics):
        velocimeter = physics.bind(self.mjcf_model.sensor.velocimeter)
        return velocimeter.sensordata + np.random.normal(0.0, 0.05, size=(3,))
    
    def get_position(self, physics : Physics):
        return physics.bind(self.root_body).xpos
    
    def set_position(self, physics : Physics, position : np.ndarray):
        physics.bind(self.root_body).xpos = position
    
    def get_framequat(self, physics : Physics) -> np.ndarray:
        return physics.bind(self.root_body).xquat
    
    def set_framequat(self, physics : Physics, quat : np.ndarray):
        physics.bind(self.root_body).xquat = quat
    
    def get_foot_positions(self, physics : Physics):
        if self._foot_sites[0] is None:
            return None
        return [physics.bind(foot).xpos for foot in self._foot_sites]

    def initialize_episode_mjcf(self, random_state):
        """Callback executed when the MJCF model is modified between episodes."""
        pass

    def after_compile(self, physics, random_state):
        """Callback executed after the Mujoco Physics is recompiled."""
        self._last_physics = physics

    def initialize_episode(self, physics, random_state):
        """Callback executed during episode initialization."""
        self._last_physics = physics
        self._last_step_position = self.get_position(physics)
        self.refresh_observation(physics)

    def before_step(self, physics, random_state):
        """Callback executed before an agent control step."""
        self._last_physics = physics

    def before_substep(self, physics, random_state):
        """Callback executed before a simulation step."""
        self._last_physics = physics

        # === Apply action ===
        if self.target_action is not None:
            if self.action_interpolation and self.prev_action is not None:
                current_qpos = physics.bind(self.observable_joints).qpos
                target_action = self.target_action
                substep_progress = min((self._substep_count + 1) * self._control_subtimestep / self._control_timestep, 1.0)
                current_target_action = (target_action - self.prev_action) * substep_progress + self.prev_action
            else:
                current_target_action = self.target_action
            self.apply_action(physics, current_target_action, random_state)
        self._substep_count += 1
        

    def after_substep(self, physics, random_state):
        """A callback which is executed after a simulation step."""
        self._last_physics = physics
        # Required for the linvel sensor.
        mujoco.mj_subtreeVel(physics.model.ptr, physics.data.ptr)
        # Update delayed observables
        for obs_name, obs_lambda in self._delayed_observable_lambdas.items():
            self._delayed_observable_queues[obs_name].append(obs_lambda(physics))

    def after_step(self, physics, random_state):
        """Callback executed after an agent control step."""
        self._last_physics = physics
    
    def refresh_observation(self,physics):
        # === Update observations read by the robot class ===
        location, quaternion = self.get_position(physics).copy(), self.get_framequat(physics).copy()
        if self._last_step_position is None:
            global_velocity = np.zeros(3)
        else:
            global_velocity = (location - self._last_step_position) / self._control_timestep + np.random.normal(0.0, 0.05, size=(3,))
        
        local_velocity = tr3d.quaternions.rotate_vector(
            global_velocity, 
            tr3d.quaternions.qinverse(quaternion)
        )

        delayed_quaternion = self.read_delayed_observables("sensors_framequat").copy()
        delayed_velocity_local = self.read_delayed_observables("sensors_local_velocimeter").copy()
        delayed_velocity_global = tr3d.quaternions.rotate_vector(delayed_velocity_local, quaternion)
        
        self.last_obs = {
            "3d_location": location, # No need to delay this since this is only used to reset the robot
            "3d_linear_velocity": delayed_velocity_global, # global_velocity,
            "3d_local_velocity": delayed_velocity_local, #local_velocity, 
            "angular_velocity": self.read_delayed_observables("sensors_gyro").copy(), # physics.bind(self.mjcf_model.sensor.gyro).sensordata.copy(),
            "framequat_wijk": delayed_quaternion, #quaternion, 
            "roll_pitch_yaw": quat_to_euler(delayed_quaternion), #quat_to_euler(quaternion), 
            "3d_acceleration_local": self.read_delayed_observables("sensors_accelerometer").copy(), # physics.bind(self.mjcf_model.sensor.accelerometer).sensordata.copy(),
            "joint_pos": self.read_delayed_observables("joints_pos").copy(), #physics.bind(self.observable_joints).qpos.copy(),
            "joint_vel": self.read_delayed_observables("joints_vel").copy(), #physics.bind(self.observable_joints).qvel.copy(),
            "joint_acc" : self.read_delayed_observables("joint_acc").copy(), #physics.bind(self.observable_joints).qacc.copy(), 
            "foot_force": self.read_delayed_observables("foot_forces").copy(), #physics.bind(self.mjcf_model.sensor.touch).sensordata.copy(),
            "torques": self.read_delayed_observables("torques").copy() #physics.bind(self.actuators).force
        }
        self._last_step_position = location
    
    def reset(self, physics):
        self.refresh_observation(physics)
        self.last_obs["3d_linear_velocity"][:] = 0
        self.last_obs["3d_local_velocity"][:] = 0
        self.last_obs["angular_velocity"][:] = 0
        self.last_obs["3d_acceleration_local"][:] = 0
        self.last_obs["joint_vel"][:] = 0
        self.last_obs["joint_acc"][:] = 0
        self.last_obs["torques"][:] = 0


"""
In compliance with `rail_walker_interface`, this class is a wrapper of the robot class.
"""
class RailSimWalkerDMControl(BaseWalker[typing.Dict[str,np.ndarray]],BaseWalkerInSim, BaseWalkerWithFootContact):
    def __init__(
        self, 
        XML_FILE : str, 
        Kp=60, 
        Kd=6, 
        action_interpolation : bool = True, 
        limit_action_range : float = 1.0, 
        power_protect_factor : float = 0.1, 
        foot_contact_threshold : np.ndarray = np.array([20.0,20.0,20.0,20.0]),
        *args, 
        **kwargs
    ):
        self.mujoco_walker = DMWalkerForRailSimWalker(XML_FILE, action_interpolation=action_interpolation, power_protect_factor=power_protect_factor, kp=Kp, kd=Kd, foot_contact_threshold=foot_contact_threshold, *args, **kwargs)
        BaseWalker.__init__(
            self,
            name="robot",
            Kp = Kp,
            Kd = Kd,
            force_real_control_timestep=False,
            limit_action_range=limit_action_range,
            power_protect_factor=power_protect_factor,
        )
        BaseWalkerInSim.__init__(self)
        BaseWalkerWithFootContact.__init__(self)


    @property
    def is_real_robot(self) -> bool:
        return False

    @property
    def action_interpolation(self):
        return self.mujoco_walker.action_interpolation
    
    @action_interpolation.setter
    def action_interpolation(self, value : bool):
        self.mujoco_walker.action_interpolation = value

    @property
    def power_protect_factor(self):
        return self.mujoco_walker.power_protect_factor
    
    @power_protect_factor.setter
    def power_protect_factor(self, value : float):
        assert value > 0 and value <= 1.0
        self.mujoco_walker.power_protect_factor = value
    
    @property
    def foot_friction(self):
        return self.mujoco_walker.foot_friction
    
    @foot_friction.setter
    def foot_friction(self, value : float):
        self.mujoco_walker.foot_friction = value

    @property
    def foot_contact_threshold(self):
        return self.mujoco_walker._built_observable._foot_contact_threshold

    @foot_contact_threshold.setter
    def foot_contact_threshold(self, value : np.ndarray):
        assert value.shape == (4,)
        self.mujoco_walker._built_observable._foot_contact_threshold = value.astype(np.float32)

    @property
    def joint_qpos_init(self) -> np.ndarray:
        pass

    @property
    def joint_qpos_offset(self) -> np.ndarray:
        return np.array([0.2, 0.4, 0.4] * 4)

    @property
    def joint_qpos_mins(self) -> np.ndarray:
        pass

    @property
    def joint_qpos_maxs(self) -> np.ndarray:
        pass

    def get_3d_location(self) -> np.ndarray:
        return self.mujoco_walker.last_obs["3d_location"]

    def set_3d_location(self, target_location : np.ndarray) -> np.ndarray:
        self.mujoco_walker.set_position(self.mujoco_walker._last_physics, target_location)
        self.mujoco_walker.refresh_observation(self.mujoco_walker._last_physics)
    
    def reset_2d_location(self, target_location : np.ndarray, target_quaternion : np.ndarray | None = None, target_qpos : np.ndarray | None = None) -> np.ndarray:
        self.reset_2d_location_with_qpos(
            target_location=target_location,
            target_quaternion=target_quaternion,
            target_qpos=target_qpos
        )

    def reset_2d_location_with_qpos(
        self,
        target_location: np.ndarray,
        target_quaternion: np.ndarray | None = None,
        target_qpos: np.ndarray | None = None,
    ):
        if target_qpos is None:
            target_qpos = self.joint_qpos_init
        find_dm_control_non_contacting_height(self.mujoco_walker._last_physics, self.mujoco_walker, target_location[0], target_location[1], qpos = target_qpos, quat = target_quaternion)
        self.mujoco_walker.target_action = target_qpos
        self.mujoco_walker.refresh_observation(self.mujoco_walker._last_physics)

    # Reset task utilities
    def reset_dropped(self, target_joints : np.ndarray, target_location : np.ndarray, target_quaternion : np.ndarray | None = None, settle : bool = False) -> np.ndarray:
        # Set robot to target joints and location and quaternion, wait for physics to settle
        physics = self.mujoco_walker._last_physics
        physics.bind(self.mujoco_walker.joints).qpos[:] = target_joints
        self.mujoco_walker.set_pose(physics, target_location, target_quaternion)
        if settle:
            # physics.bind(self.mujoco_walker.joints).qpos[:] = target_joints
            # self.mujoco_walker.set_pose(physics, target_location, target_quaternion)
            settle_physics(
                physics,
                self.mujoco_walker
            ) 
        # else:
            # find_dm_control_non_contacting_height(physics, self.mujoco_walker, target_location[0], target_location[1], qpos = target_joints, quat = target_quaternion)
            # print height of the robot
            # height = physics.bind(self.mujoco_walker.root_body).xpos[-1]
            # print(height)
        self.mujoco_walker.refresh_observation(self.mujoco_walker._last_physics)
        self.mujoco_walker.target_action = target_joints

    def set_framequat_wijk(self, framequat_wijk: np.ndarray) -> None:
        self.mujoco_walker.set_framequat(self.mujoco_walker._last_physics, framequat_wijk)
        self.mujoco_walker.refresh_observation(self.mujoco_walker._last_physics)

    def set_roll_pitch_yaw(self, roll: float, pitch: float, yaw: float) -> None:
        self.set_framequat_wijk(
            tr3d.euler.euler2quat(roll, pitch, yaw)
        )

    @property
    def control_timestep(self) -> float:
        return self.mujoco_walker._control_timestep
    
    @control_timestep.setter
    def control_timestep(self, value : float) -> None:
        self.mujoco_walker._control_timestep = value
    
    @property
    def control_subtimestep(self) -> float:
        return self.mujoco_walker._control_subtimestep

    @control_subtimestep.setter
    def control_subtimestep(self, value : float) -> None:
        self.mujoco_walker._control_subtimestep = value

    @property
    def Kp(self) -> float:
        return self.mujoco_walker.kp
    
    @Kp.setter
    def Kp(self, value : float) -> None:
        self.mujoco_walker.kp = value
    
    @property
    def Kd(self) -> float:
        return self.mujoco_walker.kd
    
    @Kd.setter
    def Kd(self, value : float) -> None:
        self.mujoco_walker.kd = value

    def receive_observation(self) -> bool:
        if self.mujoco_walker._last_physics is None:
            return False
        
        self.mujoco_walker.refresh_observation(self.mujoco_walker._last_physics)
        return self.mujoco_walker.last_obs is not None

    def reset(self) -> None:
        self.mujoco_walker.reset(self.mujoco_walker._last_physics)

    def get_3d_linear_velocity(self) -> np.ndarray:
        return self.mujoco_walker.last_obs["3d_linear_velocity"]

    def get_3d_local_velocity(self) -> np.ndarray:
        return self.mujoco_walker.last_obs["3d_local_velocity"]

    def get_3d_angular_velocity(self) -> np.ndarray:
        return self.mujoco_walker.last_obs["angular_velocity"]

    def get_framequat_wijk(self) -> np.ndarray:
        return self.mujoco_walker.last_obs["framequat_wijk"]

    def get_roll_pitch_yaw(self) -> tuple[float, float, float]:
        return self.mujoco_walker.last_obs["roll_pitch_yaw"]

    def get_last_observation(self) -> typing.Optional[typing.Dict[str,np.ndarray]]:
        return self.mujoco_walker.last_obs

    def get_3d_acceleration_local(self) -> np.ndarray:
        return self.mujoco_walker.last_obs["3d_acceleration_local"]

    def get_joint_qpos(self) -> np.ndarray:
        return self.mujoco_walker.last_obs["joint_pos"]

    def get_joint_qvel(self) -> np.ndarray:
        return self.mujoco_walker.last_obs["joint_vel"]
    
    def get_joint_qacc(self) -> np.ndarray:
        return self.mujoco_walker.last_obs["joint_acc"]

    def get_joint_torques(self) -> np.ndarray:
        return self.mujoco_walker.last_obs["torques"]

    def get_foot_force(self) -> np.ndarray:
        return self.mujoco_walker.last_obs["foot_force"]
    
    def get_foot_force_norm(self) -> np.ndarray:
        return self.get_foot_force() / 50.0
    
    def get_foot_contact(self) -> np.ndarray:
        return self.get_foot_force() >= self.foot_contact_threshold

    def _apply_action(self, action: np.ndarray) -> bool:
        self.mujoco_walker.set_target_action(action)
        return True

    def close(self) -> None:
        return