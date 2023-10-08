from rail_walker_interface import BaseWalker, BaseWalkerWithFootContact, BaseWalkerWithJointTemperatureSensor, BaseWalkerWithJoystick, Walker3DVelocityEstimator
from typing import Generic, TypeVar, Optional
import numpy as np
import unitree_go1_wrapper as go1_wrapper
from functools import cached_property
import transforms3d as tr
import multiprocessing
import threading
import time
from ctypes import c_float, c_bool

class Go1RobotThreadRunner:
    REAL_CONTROL_TIMESTEP = 0.002
    MOMENTUM_BETA = 0.95
    def __init__(
        self,
        Kp: float,
        Kd: float,
        power_protect_factor: float,
        action_interpolation: bool,
        torque_cutoff_hip: float,
        torque_cutoff_thigh: float,
        torque_cutoff_calf: float,
    ):
        assert power_protect_factor > 0 and power_protect_factor <= 1

        self.run_thread : Optional[multiprocessing.Process] = None
        self.event_req_observation : Optional[multiprocessing.Event] = None
        self.event_quit : Optional[multiprocessing.Event] = None
        self.pipe_comm : Optional[multiprocessing.Pipe] = None

        self.Kd = multiprocessing.Value(c_float, Kd)
        self.Kp = multiprocessing.Value(c_float, Kp)
        self.torque_cutoff_hip = multiprocessing.Value(c_float, torque_cutoff_hip)
        self.torque_cutoff_thigh = multiprocessing.Value(c_float, torque_cutoff_thigh)
        self.torque_cutoff_calf = multiprocessing.Value(c_float, torque_cutoff_calf)
        self.action_interpolation = multiprocessing.Value(c_bool, action_interpolation)
        self.power_protect_factor = multiprocessing.Value(c_float, power_protect_factor)
        
    def set_up_run_thread(
        self, 
        robot: "Go1RealWalker",
        control_timestep: float = 0.02,
    ):
        self.clean_up_run_thread()
        self.event_req_observation = multiprocessing.Event()
        self.event_quit = multiprocessing.Event()
        self.pipe_comm, child_pipe = multiprocessing.Pipe(duplex=True)
        self.run_thread = multiprocessing.Process(
            target=__class__.thread_run,
            args=(
                robot,
                self,
                robot.robot_interface,
                robot.robot_safety_interface,
                self.event_req_observation,
                self.event_quit,
                child_pipe,
                control_timestep,
            )
        )
        self.run_thread.start()

    @staticmethod
    def cutoff_action(
        current_action : np.ndarray, 
        current_qpos : np.ndarray, 
        current_dqpos : np.ndarray,
        max_torque : np.ndarray,
        Kp : float,
        Kd : float,
    ) -> np.ndarray:
        # Cutoff action to prevent robot from exerting too much torque on joints
        delta_qpos = current_action - current_qpos
        current_d_torque = Kd * current_dqpos
        current_d_torque = np.clip(current_d_torque, -max_torque, max_torque)
        min_delta_qpos = (current_d_torque - max_torque) / Kp
        min_delta_qpos = np.minimum(min_delta_qpos, 0) # Clip so that 0 is always a valid value
        
        max_delta_qpos = (current_d_torque + max_torque) / Kp
        max_delta_qpos = np.maximum(max_delta_qpos, 0) # Clip so that 0 is always a valid value

        clipped_delta_qpos = np.clip(delta_qpos, min_delta_qpos, max_delta_qpos)
        clipped_action = current_qpos + clipped_delta_qpos
        return clipped_action
    
    def clean_up_run_thread(self):
        if self.event_quit is not None and self.run_thread is not None:
            self.event_quit.set()
            self.run_thread.join(5.0)
            if self.run_thread.is_alive():
                self.run_thread.terminate()
            self.run_thread.close()
            self.run_thread = None

        if self.run_thread is not None:
            if self.run_thread.is_alive():
                self.run_thread.terminate()
            self.run_thread.close()
            self.run_thread = None
        
        if self.pipe_comm is not None:
            self.pipe_comm.close()
        
        self.event_quit = None
        self.event_req_observation = None
        self.pipe_comm = None
    
    def async_request_observation(self) -> None:
        if self.event_req_observation is not None:
            self.event_req_observation.set()
        else:
            raise RuntimeError("No active thread to request observation from.")
    
    def async_read_observation(self) -> Optional[tuple[go1_wrapper.LowState, np.ndarray]]:
        if self.pipe_comm is not None and self.pipe_comm.poll():
            return self.pipe_comm.recv()
        else:
            return None

    def read_observation(self) -> tuple[go1_wrapper.LowState, np.ndarray]:
        if self.event_req_observation:
            self.event_req_observation.set()
            return self.pipe_comm.recv()
        else:
            raise RuntimeError("No active thread to read observation from.")

    def async_apply_action(self, action : np.ndarray) -> None:
        if self.pipe_comm is not None:
            self.pipe_comm.send(action)
        else:
            raise RuntimeError("No active thread to apply action to.")   

    @staticmethod
    def thread_run(
        robot: BaseWalker[go1_wrapper.LowState],
        thread_runner: "Go1RobotThreadRunner",
        robot_interface: go1_wrapper.UDP,
        robot_safety_interface: go1_wrapper.Safety,
        event_req_observation: multiprocessing.Event,
        event_quit: multiprocessing.Event,
        pipe_comm: multiprocessing.Pipe,
        control_timestep: float = 0.02
    ):
        def build_command(action : np.ndarray, command : go1_wrapper.LowCmd, Kp : float, Kd : float) -> go1_wrapper.LowCmd:
            for i in range(12):
                command.motorCmd[i].q = action[i]
                command.motorCmd[i].dq = 0.0
                command.motorCmd[i].Kp = Kp
                command.motorCmd[i].Kd = Kd
                command.motorCmd[i].tau = 0.0
            return command

        thread_id = threading.get_ident()
        print(f"Go1 Robot Control Thread Started (id={thread_id}, CONTROL_TIMESTEP={__class__.REAL_CONTROL_TIMESTEP}).")
        
        try:
            robot_interface.recv()
            last_control_motor_states = [go1_wrapper.MotorState.fromNative(robot_interface._low_state_native.motorState[i]) for i in range(12)]
            last_control_qpos = np.array([last_control_motor_states[i].q for i in range(12)])
            last_control_dq = np.array([last_control_motor_states[i].dq for i in range(12)])

            last_state= None
            last_action = None
            curr_action = None
            last_control_time = 0.0
            control_command = robot_interface.initLowCmdData()
            last_action_command_t = 0.0

            power_protect_factor = thread_runner.power_protect_factor.value
            max_torque = np.array([thread_runner.torque_cutoff_hip.value, thread_runner.torque_cutoff_thigh.value, thread_runner.torque_cutoff_calf.value] * 4)
            max_torque *= power_protect_factor
            Kp = thread_runner.Kp.value
            Kd = thread_runner.Kd.value
            action_interpolation = thread_runner.action_interpolation.value

            while True:
                t = time.time()

                # Make sure we are not sending commands too fast (at REAL_CONTROL_TIMESTEP)
                action_command_dt = t - last_action_command_t
                dt = t - last_control_time
                if dt < __class__.REAL_CONTROL_TIMESTEP:
                    time.sleep(__class__.REAL_CONTROL_TIMESTEP - dt)
                    continue
                
                # check if we need to quit
                if event_quit.is_set():
                    break

                t = time.time()
                
                # Update observation / state
                robot_interface.recv()
                last_control_motor_states = [go1_wrapper.MotorState.fromNative(robot_interface._low_state_native.motorState[i]) for i in range(12)]
                last_control_qpos = np.array([last_control_motor_states[i].q for i in range(12)])
                last_control_dq = np.array([last_control_motor_states[i].dq for i in range(12)])

                # Send out observation if requested
                if event_req_observation.is_set():
                    event_req_observation.clear()
                    last_state = robot_interface.getRecvLow()
                    obs_to_send = last_state
                    pipe_comm.send(obs_to_send)

                # set last_action and last_action_command_t if we have a new action
                if pipe_comm.poll():
                    last_action = curr_action
                    curr_action = pipe_comm.recv()
                    last_action_command_t = t
                    action_command_dt = 0.0

                    # Refresh Kp, Kd, max_torque, action_interpolation
                    power_protect_factor = thread_runner.power_protect_factor.value
                    max_torque = np.array([thread_runner.torque_cutoff_hip.value, thread_runner.torque_cutoff_thigh.value, thread_runner.torque_cutoff_calf.value] * 4)
                    max_torque *= power_protect_factor
                    Kp = thread_runner.Kp.value
                    Kd = thread_runner.Kd.value
                    action_interpolation = thread_runner.action_interpolation.value

                    last_state = robot_interface.getRecvLow()
                    
                # Send out action (control)
                if curr_action is not None:
                    action_to_send = curr_action

                    if last_action is not None and action_interpolation:
                        control_progress = max(min(action_command_dt / control_timestep,1.0),0.0)
                        action_to_send = (1.0 - control_progress) * last_action + control_progress * curr_action

                    action_to_send = __class__.cutoff_action(action_to_send, last_control_qpos, last_control_dq, max_torque, Kp, Kd)
                    # print(f"Sending action: {action_to_send}")
                    control_command = build_command(action_to_send, control_command, Kp, Kd)
                    
                    # Send command & Power Protect
                    robot_interface.setToSendLow(control_command)
                    # Power protect
                    robot_safety_interface.powerProtect(10)
                    robot_safety_interface.positionLimit()
                    robot_interface.send()

        finally:
            pipe_comm.close()
            print(f"Go1 Robot Control Thread(id={thread_id}) Ended.")
    
    def __del__(self):
        self.clean_up_run_thread()


class Go1RealWalker(BaseWalker[go1_wrapper.LowState], BaseWalkerWithFootContact, BaseWalkerWithJoystick, BaseWalkerWithJointTemperatureSensor):
    def __init__(
        self, 
        velocity_estimator: Walker3DVelocityEstimator,
        power_protect_factor : float = 0.5,
        foot_contact_threshold: np.ndarray = np.array([100, 100, 100, 100]),
        action_interpolation: bool = True,
        name: Optional[str] = "robot", 
        Kp: float = 5, 
        Kd: float = 1,
        control_timestep : float = 0.05,
        force_real_control_timestep : bool = True,
        limit_action_range : float = 1.0,
        torque_cutoff_hip : float = 23.7,
        torque_cutoff_thigh : float = 23.7,
        torque_cutoff_calf : float = 35.55
    ):  
        self.thread_runner = Go1RobotThreadRunner(
            Kp=Kp, 
            Kd=Kd, 
            power_protect_factor=power_protect_factor,
            action_interpolation=action_interpolation, 
            torque_cutoff_hip=torque_cutoff_hip,
            torque_cutoff_thigh=torque_cutoff_thigh,
            torque_cutoff_calf=torque_cutoff_calf
        )
        assert control_timestep >= 0.02
        BaseWalker.__init__(self,name, Kp, Kd, force_real_control_timestep, limit_action_range, power_protect_factor)
        BaseWalkerWithFootContact.__init__(self)
        BaseWalkerWithJointTemperatureSensor.__init__(self)
        BaseWalkerWithJoystick.__init__(self)
        
        self.velocity_estimator = velocity_estimator
        self._last_state : Optional[go1_wrapper.LowState] = None
        self._last_velocity : np.ndarray = np.zeros(3)
        # self.foot_contact_threshold = foot_contact_threshold
        self.foot_contact_no_contact_threshold = np.array([36.0, 120.0, 90.0, 130.0])
        self.foot_contact_has_contact_threshold = np.array([120.0, 230.0, 340.0, 380.0])
        self.foot_contact_threshold = self.foot_contact_no_contact_threshold * 0.8 + self.foot_contact_has_contact_threshold * 0.2
        #self.foot_contact_threshold = np.array([90, 140, 150, 150])
        self.foot_contact_calibrated = True

        self._control_timestep = control_timestep
        
        self.robot_interface = go1_wrapper.UDP(go1_wrapper.ControlLevel.LOW)
        self.robot_interface.setToSendLow(self.robot_interface.initCommunicationLowCmdData())
        self.robot_interface.send()
        time.sleep(0.05) # Wait for the robot to send back the first state
        self.robot_interface.recv()

        self._last_state = self.robot_interface.getRecvLow() # Fill in _last_state just in case velocity estimator requires it
        self._last_velocity = np.zeros(3)
        self._last_velocity_estimator_t = time.time()
        self._last_joystick_values : list[np.ndarray] = [
            np.zeros(2),
            np.zeros(2)
        ]

        self.robot_safety_interface = go1_wrapper.Safety(self.robot_interface)
        
        self.thread_runner.set_up_run_thread(
            self,
            control_timestep
        )
        self.velocity_estimator.reset(self, self.get_framequat_wijk())
    
    def __copy__(self):
        raise NotImplementedError()

    def __deepcopy__(self, memo):
        raise NotImplementedError()
    
    @property
    def is_real_robot(self) -> bool:
        return True

    @property
    def control_timestep(self) -> float:
        return self._control_timestep

    @property
    def control_subtimestep(self) -> float:
        return Go1RobotThreadRunner.REAL_CONTROL_TIMESTEP
    
    @property
    def power_protect_factor(self) -> float:
        return self.thread_runner.power_protect_factor.value
    
    @power_protect_factor.setter
    def power_protect_factor(self, value: float) -> None:
        assert value > 0.0 and value <= 1.0
        self.thread_runner.power_protect_factor.value = value

    @property
    def Kp(self) -> float:
        return self.thread_runner.Kp.value
    
    @Kp.setter
    def Kp(self, value: float) -> None:
        self.thread_runner.Kp.value = value
    
    @property
    def Kd(self) -> float:
        return self.thread_runner.Kd.value
    
    @Kd.setter
    def Kd(self, value: float) -> None:
        self.thread_runner.Kd.value = value

    @property
    def torque_cutoff_hip(self) -> float:
        return self.thread_runner.torque_cutoff_hip.value
    
    @torque_cutoff_hip.setter
    def torque_cutoff_hip(self, value: float) -> None:
        self.thread_runner.torque_cutoff_hip.value = value
    
    @property
    def torque_cutoff_thigh(self) -> float:
        return self.thread_runner.torque_cutoff_thigh.value
    
    @torque_cutoff_thigh.setter
    def torque_cutoff_thigh(self, value: float) -> None:
        self.thread_runner.torque_cutoff_thigh.value = value

    @property
    def torque_cutoff_calf(self) -> float:
        return self.thread_runner.torque_cutoff_calf.value

    @torque_cutoff_calf.setter
    def torque_cutoff_calf(self, value: float) -> None:
        self.thread_runner.torque_cutoff_calf.value = value

    @property
    def action_interpolation(self) -> bool:
        return self.thread_runner.action_interpolation.value
    
    @action_interpolation.setter
    def action_interpolation(self, value: bool) -> None:
        self.thread_runner.action_interpolation.value = value

    def receive_observation(self) -> bool:
        t = time.time()
        
        # Step robot interface
        new_state = self.thread_runner.read_observation()
        is_new_state_valid = new_state.isValid()
        if not is_new_state_valid:
            if self._last_state is None:
                self._last_state = new_state
        else:
            self._last_state = new_state

        raw_joystick = go1_wrapper.XRockerBtnDataStruct.fromNative(self._last_state.wirelessRemote)
        self._last_joystick_values = [
            np.array([raw_joystick.lx, raw_joystick.ly]),
            np.array([raw_joystick.rx, raw_joystick.ry])
        ]
        
        # Step velocity estimator
        self.velocity_estimator.step(
            self,
            self.get_framequat_wijk(),
            t - self._last_velocity_estimator_t
        )
        self._last_velocity_estimator_t = t
        self._last_velocity = self.velocity_estimator.get_3d_linear_velocity()

        return is_new_state_valid

    @cached_property
    def joint_qpos_init(self) -> np.ndarray:
        real_shift = np.array([
            -0.1, -0.2, 0.0,
            0.1, -0.2, 0.0,
            -0.1, 0.0, 0.0,
            0.1, 0.0, 0.0
        ])
        # shift_delta=np.array([-0.1, -0.1, 0.1])
        # real_shift = shift_delta.repeat(4).reshape((3,4)).T.flatten()
        # real_shift[3::6] = -real_shift[3::6]
        return np.array([go1_wrapper.GO1_HIP_INIT, go1_wrapper.GO1_THIGH_INIT, go1_wrapper.GO1_CALF_INIT] * 4) + real_shift

    @cached_property
    def joint_qpos_sitting(self) -> np.ndarray:
        return np.array([0.0 / 180 * np.pi, 70.0/180*np.pi, -150.0 / 180 * np.pi] * 4)

    @cached_property
    def joint_qpos_offset(self) -> np.ndarray:
        return np.array([0.2, 0.4, 0.4] * 4)

    @cached_property
    def joint_qpos_mins(self) -> np.ndarray:
        return np.array([go1_wrapper.GO1_HIP_MIN, go1_wrapper.GO1_THIGH_MIN, go1_wrapper.GO1_CALF_MIN] * 4)

    @cached_property
    def joint_qpos_maxs(self) -> np.ndarray:
        return np.array([go1_wrapper.GO1_HIP_MAX, go1_wrapper.GO1_THIGH_MAX, go1_wrapper.GO1_CALF_MAX] * 4)

    def reset(self) -> None:
        t = time.time()
        self._last_velocity = np.zeros(3)
        self._last_velocity_estimator_t = t
        self.velocity_estimator.reset(self, self.get_framequat_wijk())
        super().reset()

    def get_3d_linear_velocity(self) -> np.ndarray:
        return self._last_velocity
    
    def get_3d_local_velocity(self) -> np.ndarray:
        try:
            if hasattr(self.velocity_estimator,"get_3d_local_velocity"):
                return self.velocity_estimator.get_3d_local_velocity()
            else:
                return tr.quaternions.rotate_vector(
                    self.get_3d_linear_velocity(), 
                    tr.quaternions.qinverse(self.get_framequat_wijk())
                )
        except:
            print("Error in get_3d_local_velocity, continuing anyway")
            import traceback
            traceback.print_exc()
            return np.zeros(3)

    def get_3d_angular_velocity(self) -> np.ndarray:
        return self._last_state.imu.gyroscope
    
    def get_framequat_wijk(self) -> np.ndarray:
        return self._last_state.imu.quaternion
    
    def get_roll_pitch_yaw(self) -> np.ndarray:
        return self._last_state.imu.rpy

    def get_last_observation(self) -> Optional[go1_wrapper.LowState]:
        return self._last_state

    def get_3d_acceleration_local(self) -> np.ndarray:
        return self._last_state.imu.accelerometer

    def get_joint_qpos(self) -> np.ndarray:
        return np.array([ms.q for ms in self._last_state.motorState[:12]])

    def get_joint_qvel(self) -> np.ndarray:
        return np.array([ms.dq for ms in self._last_state.motorState[:12]])
    
    def get_joint_qacc(self) -> np.ndarray:
        return np.array([ms.ddq for ms in self._last_state.motorState[:12]])

    def get_joint_torques(self) -> np.ndarray:
        return np.array([ms.tauEst for ms in self._last_state.motorState[:12]])
    
    def get_joint_temperature_celsius(self) -> np.ndarray:
        return np.array([ms.temperature for ms in self._last_state.motorState[:12]], dtype=np.float32)
    
    def hibernate(self) -> None:
        if self.thread_runner.run_thread is not None:
            print("Hibernating...")
            temp_action_range = self.limit_action_range # Save action range
            self.limit_action_range = 1.0 # Set action range to 1.0 temporarily
            start_qpos = self.get_joint_qpos()
            delta_qpos = self.joint_qpos_sitting - start_qpos
            for i in range(20):
                self.apply_action(start_qpos + delta_qpos * ((i+1) / 20.0))
            self.limit_action_range = temp_action_range # Restore action range

        self.thread_runner.clean_up_run_thread()
    
    def cancel_hibernate(self) -> None:
        self.robot_interface.setToSendLow(self.robot_interface.initCommunicationLowCmdData())
        self.robot_interface.send()
        time.sleep(0.05) # Wait for the robot to send back the first state
        self.robot_interface.recv()

        self._last_state = self.robot_interface.getRecvLow() # Fill in _last_state just in case velocity estimator requires it
        self._last_velocity = np.zeros(3)
        self._last_velocity_estimator_t = time.time()
        self.thread_runner.set_up_run_thread(self,self.control_timestep)

    def _apply_action(self, action: np.ndarray) -> bool:
        # if np.allclose(action, self.joint_qpos_init):
        #     print("Applying Standing Pose")
        # else:
        #     print("Applying Action", action)
        
        self.thread_runner.async_apply_action(action)
        return True
    
    def get_foot_contact(self) -> np.ndarray:
        return self.get_foot_force() >= self.foot_contact_threshold
    
    def get_foot_force(self) -> np.ndarray:
        return self._last_state.footForce
    
    def get_foot_force_norm(self) -> np.ndarray:
        return (self.get_foot_force() - self.foot_contact_no_contact_threshold) / (self.foot_contact_has_contact_threshold - self.foot_contact_no_contact_threshold)
    
    def get_joystick_values(self) -> list[np.ndarray]:
        return self._last_joystick_values
    
    def close(self) -> None:
        self.thread_runner.clean_up_run_thread()
        self.velocity_estimator.close()