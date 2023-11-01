# https://github.com/unitreerobotics/unitree_legged_sdk/blob/go1/include/unitree_legged_sdk/udp.h
from .native_interface import robot_interface
from . import common_constants, comm
import copy

UDP_CLIENT_PORT = 8080
UDP_SERVER_PORT_BASIC = 8007
UDP_SERVER_PORT_SPORT = 8082
UDP_SERVER_IP_BASIC = "192.168.123.10"
UDP_SERVER_IP_SPORT = "192.168.123.161"

class UDP:
    """
    Wrapper for robot_interface.UDP
    """

    def __init__(
            self,
            control_level: common_constants.ControlLevel, 
            #localPort : int = UDP_CLIENT_PORT, 
            #target_ip : str = UDP_SERVER_IP_BASIC, 
            #target_port : int = UDP_SERVER_PORT_BASIC
        ):
        #self._udp = robot_interface.UDP(control_level.value, localPort, target_ip, target_port)
        if control_level == common_constants.ControlLevel.LOW:
            self._udp = robot_interface.UDP(control_level.value, UDP_CLIENT_PORT, UDP_SERVER_IP_BASIC, UDP_SERVER_PORT_BASIC)
            self._low_state_native = robot_interface.LowState()
            self._low_cmd_native = robot_interface.LowCmd()
            self._udp.InitCmdData(self._low_cmd_native)
            self._low_cmd_default = comm.LowCmd.fromNative(self._low_cmd_native)
            self._low_state = None
        elif control_level == common_constants.ControlLevel.HIGH:
            self._udp = robot_interface.UDP(control_level.value, UDP_CLIENT_PORT, UDP_SERVER_IP_SPORT, UDP_SERVER_PORT_SPORT)
            self._high_state_native = robot_interface.HighState()
            self._high_cmd_native = robot_interface.HighCmd()
            self._udp.InitCmdData(self._high_cmd_native)
            self._high_cmd_default = comm.HighCmd.fromNative(self._high_cmd_native)
            self._high_state = None
        self.control_level = control_level
    
    def setRecvTimeout(self,timeInMs : int) -> None:
        """
        Sets the timeout for receiving data from the UDP server
        """
        self._udp.setRecvTimeout(timeInMs)
    
    def send(self) -> None:
        """
        Sends data waiting in the client buffer to the server
        """
        if self.control_level == common_constants.ControlLevel.LOW:
            self._udp.SetSend(self._low_cmd_native)
        else:
            self._udp.SetSend(self._high_cmd_native)
        self._udp.Send()

    def recv(self) -> None:
        """
        Receives data from UDP server and puts it in a temporary buffer inside the UDP client. 
        The data saved in the buffer can be retrieved by calling getRecvLow() or getRecvHigh()
        """
        self._udp.Recv()
        if self.control_level == common_constants.ControlLevel.LOW:
            self._udp.GetRecv(self._low_state_native)
            self._low_state = None # clear conversion cache
        else:
            self._udp.GetRecv(self._high_state_native)
            self._high_state = None # clear conversion cache
    
    def initLowCmdData(self) -> comm.LowCmd:
        """
        Returns an initialized LowCmd with default values
        """
        return comm.LowCmd(
            copy.copy(self._low_cmd_default.head),
            copy.copy(self._low_cmd_default.version),
            self._low_cmd_default.bandWidth,
            [copy.copy(self._low_cmd_default.motorCmd[i]) for i in range(len(self._low_cmd_default.motorCmd))],
            copy.copy(self._low_cmd_default.bms),
            self._low_cmd_default.wirelessRemote.copy(),
            self._low_cmd_default.crc
        )
    
    def initCommunicationLowCmdData(self) -> comm.LowCmd:
        init_cmd = self.initLowCmdData()
        for i in range(12):
            init_cmd.motorCmd[i].q = common_constants.POS_STOP_F
            init_cmd.motorCmd[i].dq = common_constants.VEL_STOP_F
            init_cmd.motorCmd[i].Kp = 0.0
            init_cmd.motorCmd[i].Kd = 0.0
            init_cmd.motorCmd[i].tau = 0.0
        return init_cmd

    
    def initHighCmdData(self) -> comm.HighCmd:
        """
        Takes in a pointer to a HighCmd struct and initializes it with default values
        """
        return comm.HighCmd(
            copy.copy(self._high_cmd_default.head),
            self._high_cmd_default.levelFlag,
            self._high_cmd_default.frameReserve,
            copy.copy(self._high_cmd_default.SN),
            copy.copy(self._high_cmd_default.version),
            self._high_cmd_default.bandWidth,
            self._high_cmd_default.mode,
            self._high_cmd_default.gaitType,
            self._high_cmd_default.speedLevel,
            self._high_cmd_default.footRaiseHeight,
            self._high_cmd_default.bodyHeight,
            self._high_cmd_default.position.copy(),
            self._high_cmd_default.euler.copy(),
            self._high_cmd_default.velocity.copy(),
            self._high_cmd_default.yawSpeed,
            copy.copy(self._high_cmd_default.bms),
            [copy.copy(self._high_cmd_default.led[i] for i in range(len(self._high_cmd_default.led)))],
            self._high_cmd_default.wirelessRemote.copy(),
            self._high_cmd_default.crc
        )
    
    def setToSendLow(self, lowCmd : comm.LowCmd) -> None:
        """
        Takes in a pointer to a LowCmd struct and sets it to be sent later
        To actually send this data after calling this function, call send()
        """
        self._low_cmd_native = lowCmd.toNativeM(self._low_cmd_native)

    def setToSendHigh(self, highCmd : comm.HighCmd) -> None:
        """
        Takes in a pointer to a HighCmd struct and sets it to be sent later
        To actually send this data after calling this function, call send()
        """
        self._high_cmd_native = highCmd.toNativeM(self._high_cmd_native)
    
    def getRecvLow(self) -> comm.LowState:
        """
        Constructs a LowState struct and fills it with the data received from the server
        """
        if self._low_state is None:
            self._low_state = comm.LowState.fromNative(self._low_state_native)
        return self._low_state
    
    def getRecvHigh(self) -> comm.HighState:
        """
        Constructs a HighState struct and fills it with the data received from the server
        """
        if self._high_state is None:
            self._high_state = comm.HighState.fromNative(self._high_state_native)
        return self._high_state
        

