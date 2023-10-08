import numpy as np
from rail_walker_interface import JoystickPolicyResetter, JoystickPolicyTerminationConditionProvider
from rail_walker_interface import BaseWalker, BaseWalkerWithFootContact, BaseWalkerWithJoystick
from typing import Any
import time

class JoystickPolicyManualResetter(JoystickPolicyResetter[BaseWalker]):
    def __init__(self, target_pose : str = "standing", max_seconds : float = 1.0, offset_factor : float = 0.02):
        assert target_pose in ['standing', 'sitting']
        assert max_seconds > 0
        assert offset_factor >= 0 and offset_factor <= 1
        
        self.target_pose = target_pose
        self.max_seconds = max_seconds
        self.offset_factor = offset_factor
        super().__init__()
    
    def perform_reset(
        self, 
        Robot: BaseWalker, 
        info_dict: dict[str,Any], 
        termination_provider_triggered: JoystickPolicyTerminationConditionProvider,
        randomState: np.random.RandomState
    ) -> None:
        print()
        print("=================== Manual Reset ===================")
        if isinstance(Robot.unwrapped(), BaseWalkerWithJoystick):
            print("Lift up the robot and press down the right joystick to continue")
            if hasattr(Robot, "foot_contact_no_contact_threshold") and hasattr(Robot, "foot_contact_has_contact_threshold") and not hasattr(Robot, "foot_contact_calibrated"):
                print("Calibrating foot contact thresholds")
            while True:
                Robot.receive_observation()
                if Robot.get_joystick_values()[-1][1] < -0.5:
                    break
                time.sleep(Robot.control_timestep)
            if hasattr(Robot, "foot_contact_no_contact_threshold") and hasattr(Robot, "foot_contact_has_contact_threshold") and not hasattr(Robot, "foot_contact_calibrated"):
                Robot.foot_contact_no_contact_threshold = Robot.get_foot_force()
        else:
            if hasattr(Robot, "foot_contact_no_contact_threshold") and hasattr(Robot, "foot_contact_has_contact_threshold") and not hasattr(Robot, "foot_contact_calibrated"):
                input("Lift up the robot and press Enter to continue")
                Robot.receive_observation()
                Robot.foot_contact_no_contact_threshold = Robot.get_foot_force()
            else:
                input("Lift up the robot or place the robot on the ground and press Enter to continue")
            
        
        print("Resetting robot")
        print()

        target_qpos = Robot.joint_qpos_sitting if self.target_pose == "sitting" else Robot.joint_qpos_init
        Reset_Min_QPos = target_qpos - (target_qpos - Robot.joint_qpos_mins) * self.offset_factor
        Reset_Max_QPos = target_qpos + (Robot.joint_qpos_maxs - target_qpos) * self.offset_factor

        start_t = time.time()
        t = start_t

        Robot.receive_observation()
        start_qpos = Robot.get_joint_qpos()
        delta_qpos = target_qpos - start_qpos
        
        while (t - start_t <= self.max_seconds):
            # print(f"Resetting robot trial {step}")
            progress = (t - start_t) / self.max_seconds * 2.0 # Take half of the time to move to the target pose
            progress = min(progress, 1.0)
            Robot.apply_action(start_qpos + delta_qpos * progress)
            if Robot.receive_observation():
                current_qpos = Robot.get_joint_qpos()
                if np.all(current_qpos >= Reset_Min_QPos) and np.all(current_qpos <= Reset_Max_QPos):
                    print("Reset successful")
                    break
            t = time.time()
        
        if isinstance(Robot.unwrapped(), BaseWalkerWithFootContact):
            print("Finished Foot Force: ", Robot.get_foot_force())
        
        if isinstance(Robot.unwrapped(), BaseWalkerWithJoystick):
            print("Finished, press down the right joystick to continue")
            while True:
                Robot.receive_observation()
                if Robot.get_joystick_values()[-1][1] < -0.5:
                    break
                time.sleep(Robot.control_timestep)
        else:
            input("Finished, press Enter to continue")
        
        if hasattr(Robot, "foot_contact_no_contact_threshold") and hasattr(Robot, "foot_contact_has_contact_threshold") and not hasattr(Robot, "foot_contact_calibrated"):
            Robot.foot_contact_has_contact_threshold = Robot.get_foot_force()
            Robot.foot_contact_calibrated = True
            if hasattr(Robot, "foot_contact_threshold"):
                try:
                    setattr(Robot, "foot_contact_threshold", (Robot.foot_contact_has_contact_threshold + Robot.foot_contact_no_contact_threshold)/2.0)
                except:
                    pass
        
        print("====================================================")


