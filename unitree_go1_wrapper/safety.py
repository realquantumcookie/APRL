# https://github.com/unitreerobotics/unitree_legged_sdk/blob/go1/include/unitree_legged_sdk/safety.h
from .native_interface import robot_interface
from . import comm, common_constants
from .udp import UDP

class Safety:
    """
    Safety is a class that provides safety functions to call in each robot control loop
    When a command to be issued to the robot exceeds a certain safety limit, the function will halt the entire program
    """
    
    def __init__(self, udp: UDP):
        self._safety = robot_interface.Safety(robot_interface.LeggedType.Go1)
        self.udp = udp
    
    def positionLimit(self) -> None:
        if self.udp.control_level == common_constants.ControlLevel.LOW:
            self._safety.PositionLimit(self.udp._low_cmd_native)
    
    def powerProtect(self, factor : int) -> None:
        """
        Apply percentage power protection
        factor ranges in [1,10] meaning 10% to 100% repsectively
        """
        if self.udp.control_level == common_constants.ControlLevel.LOW:
            self._safety.PowerProtect(self.udp._low_cmd_native, self.udp._low_state_native, factor)
    
    def positionProtect(self, limit : float = 0.087) -> None:
        """
        limit is the maximum angle in radians, default is 5 deg
        """
        if self.udp.control_level == common_constants.ControlLevel.LOW:
            self._safety.PositionProtect(self.udp._low_cmd_native, self.udp._low_state_native, limit)