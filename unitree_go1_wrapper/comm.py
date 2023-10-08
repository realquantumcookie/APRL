# https://github.com/unitreerobotics/unitree_legged_sdk/blob/go1/include/unitree_legged_sdk/comm.h
from .native_interface import robot_interface
import typing
import numpy as np

class HasNativeRepr:
    def toNative(self):
        pass

    def toNativeM(self, native):
        pass

    @staticmethod
    def fromNative(native):
        pass

    def __repr__(self) -> str:
        return str(self.__dict__)

class State:
    def isValid(self):
        raise NotImplementedError()

class BmsCmd(HasNativeRepr):
    def __init__(self, off: int = 0xa5):
        self.off = off #off 0xA5
    
    def toNative(self):
        nat = robot_interface.BmsCmd()
        return self.toNativeM(nat)

    def toNativeM(self, native):
        native.off = self.off
        return native
    
    @staticmethod
    def fromNative(native):
        return __class__(native.off)

class BmsState(HasNativeRepr, State):
    def __init__(
        self, 
        version_h : int, #uint8_t
        version_l : int, #uint8_t
		bms_status : int, #uint8_t
		SOC : int, #SOC 0-100%, uint8_t
		current : int, #mA, int32_t
		cycle : int, #uint16_t
		BQ_NTC : typing.Tuple[int,int], #x1 degrees centigrade, std::array<int8_t, 2>
		MCU_NTC: typing.Tuple[int,int], # x1 degrees centigrade, std::array<int8_t, 2>
		cell_vol : typing.Tuple[int,int,int,int,int,int,int,int,int,int] #cell Voltage mV, std::array<uint16_t, 10> 
    ):
        assert len(BQ_NTC) == 2 and len(MCU_NTC) == 2 and len(cell_vol) == 10

        self.version_h = version_h
        self.version_l = version_l
        self.bms_status = bms_status
        self.SOC = SOC
        self.current = current
        self.cycle = cycle
        self.BQ_NTC = BQ_NTC
        self.MCU_NTC = MCU_NTC
        self.cell_vol = cell_vol
    
    def toNative(self):
        assert len(self.BQ_NTC) == 2 and len(self.MCU_NTC) == 2 and len(self.cell_vol) == 10

        nat = robot_interface.BmsState()
        return self.toNativeM(nat)

    def toNativeM(self, native):
        native.version_h = self.version_h
        native.version_l = self.version_l
        native.bms_status = self.bms_status
        native.SOC = self.SOC
        native.current = self.current
        native.cycle = self.cycle
        for i in range(2):
            native.BQ_NTC[i] = self.BQ_NTC[i]
            native.MCU_NTC[i] = self.MCU_NTC[i]
        for i in range(10):
            native.cell_vol[i] = self.cell_vol[i]
        return native
    
    @staticmethod
    def fromNative(native):
        return __class__(
            native.version_h,
            native.version_l,
            native.bms_status,
            native.SOC,
            native.current,
            native.cycle,
            (native.BQ_NTC[0], native.BQ_NTC[1]),
            (native.MCU_NTC[0], native.MCU_NTC[1]),
            (native.cell_vol[0], native.cell_vol[1], native.cell_vol[2], native.cell_vol[3], native.cell_vol[4], native.cell_vol[5], native.cell_vol[6], native.cell_vol[7], native.cell_vol[8], native.cell_vol[9])
        )

    def isValid(self):
        return (
            self.version_h != 0x0 or
            self.version_l != 0x0 or
            self.bms_status != 0x0 or
            self.SOC != 0x0 or
            self.current != 0x0 or
            self.cycle != 0x0 or
            self.BQ_NTC != (0,0) or
            self.MCU_NTC != (0,0) or
            self.cell_vol != (0,0,0,0,0,0,0,0,0,0)
        )

class Cartesian(HasNativeRepr, State):
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z
    
    def toNative(self):
        nat = robot_interface.Cartesian()
        return self.toNativeM(nat)

    def toNativeM(self, native):
        native.x = self.x
        native.y = self.y
        native.z = self.z
        return native
    
    @staticmethod
    def fromNative(native):
        return __class__(native.x, native.y, native.z)

    def isValid(self):
        return self.x != 0 or self.y != 0 or self.z != 0

class IMU(HasNativeRepr, State):
    def __init__(
        self,
        quaternion: np.ndarray,
        gyroscope: np.ndarray,
        accelerometer: np.ndarray,
        rpy: np.ndarray,
        temperature: int
    ) -> None:
        assert quaternion.shape == (4,)
        assert gyroscope.shape == (3,)
        assert accelerometer.shape == (3,)
        assert rpy.shape == (3,)
        self.quaternion = quaternion
        self.gyroscope = gyroscope
        self.accelerometer = accelerometer
        self.rpy = rpy
        self.temperature = temperature
    
    def toNative(self):
        nat = robot_interface.IMU()
        return self.toNativeM(nat)
    
    def toNativeM(self, native):
        assert self.quaternion.shape == (4,)
        assert self.gyroscope.shape == (3,)
        assert self.accelerometer.shape == (3,)
        assert self.rpy.shape == (3,)

        for i in range(4):
            native.quaternion[i] = self.quaternion[i]
        for i in range(3):
            native.gyroscope[i] = self.gyroscope[i]
            native.accelerometer[i] = self.accelerometer[i]
            native.rpy[i] = self.rpy[i]
        native.temperature = self.temperature
        return native

    @staticmethod
    def fromNative(native):
        return __class__(
            np.array([native.quaternion[0], native.quaternion[1], native.quaternion[2], native.quaternion[3]]),
            np.array([native.gyroscope[0], native.gyroscope[1], native.gyroscope[2]]),
            np.array([native.accelerometer[0], native.accelerometer[1], native.accelerometer[2]]),
            np.array([native.rpy[0], native.rpy[1], native.rpy[2]]),
            native.temperature
        )

    def isValid(self):
        return (
            np.any(self.quaternion != 0) or
            np.any(self.gyroscope != 0) or
            np.any(self.accelerometer != 0) or
            np.any(self.rpy != 0) or
            self.temperature != 0
        )

class LED(HasNativeRepr, State):
    def __init__(
        self,
        r: int,
        g: int,
        b: int
    ) -> None:
        assert r >= 0 and r <= 255
        assert g >= 0 and g <= 255
        assert b >= 0 and b <= 255

        self.r = r
        self.g = g
        self.b = b
    
    def toNative(self):
        nat = robot_interface.LED()
        return self.toNativeM(nat)

    def toNativeM(self, native):
        # assert self.r >= 0 and self.r <= 255
        # assert self.g >= 0 and self.g <= 255
        # assert self.b >= 0 and self.b <= 255
        native.r = self.r
        native.g = self.g
        native.b = self.b
        return native
    
    @staticmethod
    def fromNative(native):
        return __class__(native.r, native.g, native.b)
    
    def isValid(self):
        return self.r != 0 or self.g != 0 or self.b != 0
    
class MotorState(HasNativeRepr, State):
    def __init__(
        self,
        mode: int, #motor working mode Servo : 0x0A, Damping : 0x00，Overheat ： 0x08.
        q: float, #current angle(in radian)
        dq: float, #current velocity(in radian/second)
        ddq: float, #current acceleration (in radian/second/second)
        tauEst: float, #current estimated output torque (N*m)
        q_raw: float,
        dq_raw : float,
        ddq_raw : float,
        temperature: int #current temperature (temperature conduction is slow that leads to lag)
    ):
        self.mode = mode
        self.q = q
        self.dq = dq
        self.ddq = ddq
        self.tauEst = tauEst
        self.q_raw = q_raw
        self.dq_raw = dq_raw
        self.ddq_raw = ddq_raw
        self.temperature = temperature
    
    def toNative(self):
        nat = robot_interface.MotorState()
        return self.toNativeM(nat)
    
    def toNativeM(self, native):
        native.mode = self.mode
        native.q = self.q
        native.dq = self.dq
        native.ddq = self.ddq
        native.tauEst = self.tauEst
        native.q_raw = self.q_raw
        native.dq_raw = self.dq_raw
        native.ddq_raw = self.ddq_raw
        native.temperature = self.temperature
        return native
    
    @staticmethod
    def fromNative(native):
        return __class__(
            native.mode,
            native.q,
            native.dq,
            native.ddq,
            native.tauEst,
            native.q_raw,
            native.dq_raw,
            native.ddq_raw,
            native.temperature
        )

    def isValid(self):
        return (
            self.mode != 0 or
            self.q != 0 or
            self.dq != 0 or
            self.ddq != 0 or
            self.tauEst != 0 or
            self.q_raw != 0 or
            self.dq_raw != 0 or
            self.ddq_raw != 0 or
            self.temperature != 0
        )

class MotorCmd(HasNativeRepr):
    def __init__(
        self,
        mode: int, #motor working mode Servo : 0x0A, Damping : 0x00.
        q: float, #target angle(in radian)
        dq: float, #target velocity(in radian/second)
        tau: float, #target output torque (N*m)
        Kp: float, #desired position stiffness (unit: N.m/rad )
        Kd: float #desired velocity stiffness (unit: N.m/(rad/s) 
    ) -> None:
        self.mode = mode
        self.q = q
        self.dq = dq
        self.tau = tau
        self.Kp = Kp
        self.Kd = Kd
    
    def toNative(self):
        nat = robot_interface.MotorCmd()
        return self.toNativeM(nat)
    
    def toNativeM(self, native):
        native.mode = self.mode
        native.q = self.q
        native.dq = self.dq
        native.tau = self.tau
        native.Kp = self.Kp
        native.Kd = self.Kd
        return native
    
    @staticmethod
    def fromNative(native):
        return __class__(
            native.mode,
            native.q,
            native.dq,
            native.tau,
            native.Kp,
            native.Kd
        )

class LowState(HasNativeRepr, State):
    def __init__(
        self,
        head: typing.Tuple[int, int],
        levelFlag: int,
        frameReserve: int,
        SN: typing.Tuple[int, int],
        version: typing.Tuple[int, int],
        bandWidth: int,
        imu: IMU,
        motorState: typing.List[MotorState],
        bms: BmsState,
        footForce: np.ndarray,
        footForceEst: np.ndarray,
        tick: int,
        wirelessRemote: np.ndarray,
        crc: int
    ):
        assert len(motorState) == 20
        assert footForce.shape == (4,)
        assert footForceEst.shape == (4,)
        assert wirelessRemote.shape == (40,) and wirelessRemote.dtype == np.uint8

        self.head = head
        self.levelFlag = levelFlag
        self.frameReserve = frameReserve
        self.SN = SN
        self.version = version
        self.bandWidth = bandWidth
        self.imu = imu
        self.motorState = motorState
        self.bms = bms
        self.footForce = footForce
        self.footForceEst = footForceEst
        self.tick = tick
        self.wirelessRemote = wirelessRemote
        self.crc = crc

    def toNative(self):
        nat = robot_interface.LowState()
        return self.toNativeM(nat)
    
    def toNativeM(self, native):
        assert len(self.motorState) == 20
        assert self.footForce.shape == (4,)
        assert self.footForceEst.shape == (4,)
        assert self.wirelessRemote.shape == (40,)

        native.head[0] = self.head[0]
        native.head[1] = self.head[1]
        native.levelFlag = self.levelFlag
        native.frameReserve = self.frameReserve
        native.SN[0] = self.SN[0]
        native.SN[1] = self.SN[1]
        native.version[0] = self.version[0]
        native.version[1] = self.version[1]
        native.bandWidth = self.bandWidth
        #native.imu = self.imu.toNative()
        self.imu.toNativeM(native.imu)
        for i in range(20):
            #native.motorState[i] = self.motorState[i].toNative()
            self.motorState[i].toNativeM(native.motorState[i])
        #native.bms = self.bms.toNative()
        self.bms.toNativeM(native.bms)
        for i in range(4):
            native.footForce[i] = self.footForce[i]
        for i in range(4):
            native.footForceEst[i] = self.footForceEst[i]
        native.tick = self.tick
        for i in range(40):
            native.wirelessRemote[i] = self.wirelessRemote[i]
        native.crc = self.crc
        return native

    @staticmethod
    def fromNative(native):
        return __class__(
            (native.head[0], native.head[1]),
            native.levelFlag,
            native.frameReserve,
            (native.SN[0], native.SN[1]),
            (native.version[0], native.version[1]),
            native.bandWidth,
            IMU.fromNative(native.imu),
            [MotorState.fromNative(native.motorState[i]) for i in range(20)],
            BmsState.fromNative(native.bms),
            np.asarray(native.footForce),
            np.asarray(native.footForceEst),
            native.tick,
            np.asarray(native.wirelessRemote, dtype=np.uint8),
            native.crc
        )

    def isValid(self):
        return (
            self.head[0] != 0 or
            self.head[1] != 0 or
            self.levelFlag != 0 or
            self.frameReserve != 0 or
            self.SN[0] != 0 or
            self.SN[1] != 0 or
            self.version[0] != 0 or
            self.version[1] != 0 or
            self.bandWidth != 0 or
            self.imu.isValid() or
            any([m.isValid() for m in self.motorState]) or
            self.bms.isValid() or
            np.any(self.footForce != 0) or
            np.any(self.footForceEst != 0) or
            self.tick != 0 or
            np.any(self.wirelessRemote != 0) or
            self.crc != 0
        )
    
class LowCmd(HasNativeRepr):
    def __init__(
        self,
        head: typing.Tuple[int, int],
        version: typing.Tuple[int, int],
        bandWidth: int,
        motorCmd: typing.List[MotorCmd],
        bms: BmsCmd,
        wirelessRemote: np.ndarray,
        crc: int
    ):
        assert len(motorCmd) == 12
        assert wirelessRemote.shape == (40,) and wirelessRemote.dtype == np.uint8

        self.head = head
        self.version = version
        self.bandWidth = bandWidth
        self.motorCmd = motorCmd
        self.bms = bms
        self.wirelessRemote = wirelessRemote
        self.crc = crc
    
    def toNative(self):
        nat = robot_interface.LowCmd()
        return self.toNativeM(nat)
    
    def toNativeM(self, native):
        # assert len(self.motorCmd) == 20
        # assert self.wirelessRemote.shape == (40,)

        native.head[0] = self.head[0]
        native.head[1] = self.head[1]
        native.version[0] = self.version[0]
        native.version[1] = self.version[1]
        native.bandWidth = self.bandWidth
        for i in range(len(self.motorCmd)):
            #native.motorCmd[i] = self.motorCmd[i].toNative()
            self.motorCmd[i].toNativeM(native.motorCmd[i])
        #native.bms = self.bms.toNative()
        self.bms.toNativeM(native.bms)
        for i in range(40):
            native.wirelessRemote[i] = self.wirelessRemote[i]
        native.crc = self.crc
        return native
    
    @staticmethod
    def fromNative(native):
        return __class__(
            (native.head[0], native.head[1]),
            (native.version[0], native.version[1]),
            native.bandWidth,
            [MotorCmd.fromNative(native.motorCmd[i]) for i in range(12)], # 20 in the native struct
            BmsCmd.fromNative(native.bms),
            np.asarray(native.wirelessRemote, dtype=np.uint8),
            native.crc
        )

class HighState(HasNativeRepr, State):
    def __init__(
        self,
        head: typing.Tuple[int, int],
        levelFlag: int,
        frameReserve: int,
        SN: typing.Tuple[int, int],
        version: typing.Tuple[int, int],
        bandWidth: int,
        imu: IMU,
        motorState: typing.List[MotorState],
        bms: BmsState,
        footForce: np.ndarray,
        footForceEst: np.ndarray,
        mode: int,
        progress: int,
        gaitType: int, # 0.idle  1.trot  2.trot running  3.climb stair  4.trot obstacle
        footRaiseHeight: float, #(unit: m, default: 0.08m), foot up height while walking
        position: np.ndarray, #(unit: m), from own odometry in inertial frame, usually drift
        bodyHeight: float, #(unit: m, default: 0.28m)
        velocity: np.ndarray, #(unit: m/s), forwardSpeed, sideSpeed, rotateSpeed in body frame
        yawSpeed: float, #in rad/s,  rotateSpeed in body frame
        rangeObstacle: np.ndarray,
        footPosition2Body: typing.List[Cartesian], #foot position relative to body
        footSpeed2Body: typing.List[Cartesian], #foot speed relative to body
        wirelessRemote: np.ndarray,
        crc: int
    ):
        assert len(motorState) == 20
        assert footForce.shape == (4,)
        assert footForceEst.shape == (4,)
        assert position.shape == (3,)
        assert velocity.shape == (3,)
        assert rangeObstacle.shape == (4,)
        assert len(footPosition2Body) == 4
        assert len(footSpeed2Body) == 4
        assert wirelessRemote.shape == (40,)

        self.head = head
        self.levelFlag = levelFlag
        self.frameReserve = frameReserve
        self.SN = SN
        self.version = version
        self.bandWidth = bandWidth
        self.imu = imu
        self.motorState = motorState
        self.bms = bms
        self.footForce = footForce
        self.footForceEst = footForceEst
        self.mode = mode
        self.progress = progress
        self.gaitType = gaitType
        self.footRaiseHeight = footRaiseHeight
        self.position = position
        self.bodyHeight = bodyHeight
        self.velocity = velocity
        self.yawSpeed = yawSpeed
        self.rangeObstacle = rangeObstacle
        self.footPosition2Body = footPosition2Body
        self.footSpeed2Body = footSpeed2Body
        self.wirelessRemote = wirelessRemote
        self.crc = crc
    
    def toNative(self):
        nat = robot_interface.HighState()
        return self.toNativeM(nat)
    
    def toNativeM(self, native):
        # assert len(self.motorState) == 20
        # assert self.footForce.shape == (4,)
        # assert self.footForceEst.shape == (4,)
        # assert self.position.shape == (3,)
        # assert self.velocity.shape == (3,)
        # assert self.rangeObstacle.shape == (4,)
        # assert len(self.footPosition2Body) == 4
        # assert len(self.footSpeed2Body) == 4
        # assert self.wirelessRemote.shape == (40,)

        native.head[0] = self.head[0]
        native.head[1] = self.head[1]
        native.levelFlag = self.levelFlag
        native.frameReserve = self.frameReserve
        native.SN[0] = self.SN[0]
        native.SN[1] = self.SN[1]
        native.version[0] = self.version[0]
        native.version[1] = self.version[1]
        native.bandWidth = self.bandWidth
        #native.imu = self.imu.toNative()
        self.imu.toNativeM(native.imu)
        for i in range(20):
            #native.motorState[i] = self.motorState[i].toNative()
            self.motorState[i].toNativeM(native.motorState[i])
        #native.bms = self.bms.toNative()
        self.bms.toNativeM(native.bms)
        for i in range(4):
            native.footForce[i] = self.footForce[i]
        for i in range(4):
            native.footForceEst[i] = self.footForceEst[i]
        native.mode = self.mode
        native.progress = self.progress
        native.gaitType = self.gaitType
        native.footRaiseHeight = self.footRaiseHeight
        for i in range(3):
            native.position[i] = self.position[i]
        native.bodyHeight = self.bodyHeight
        for i in range(3):
            native.velocity[i] = self.velocity[i]
        native.yawSpeed = self.yawSpeed
        for i in range(8):
            native.rangeObstacle[i] = self.rangeObstacle[i]
        for i in range(4):
            #native.footPosition2Body[i] = self.footPosition2Body[i].toNative()
            self.footPosition2Body[i].toNativeM(native.footPosition2Body[i])
        for i in range(4):
            #native.footSpeed2Body[i] = self.footSpeed2Body[i].toNative()
            self.footSpeed2Body[i].toNativeM(native.footSpeed2Body[i])
        for i in range(40):
            native.wirelessRemote[i] = self.wirelessRemote[i]
        native.crc = self.crc
        return native
    
    @staticmethod
    def fromNative(native):
        return __class__(
            (native.head[0], native.head[1]),
            native.levelFlag,
            native.frameReserve,
            (native.SN[0], native.SN[1]),
            (native.version[0], native.version[1]),
            native.bandWidth,
            IMU.fromNative(native.imu),
            [MotorState.fromNative(native.motorState[i]) for i in range(20)],
            BmsState.fromNative(native.bms),
            np.array([native.footForce[i] for i in range(4)]),
            np.array([native.footForceEst[i] for i in range(4)]),
            native.mode,
            native.progress,
            native.gaitType,
            native.footRaiseHeight,
            np.array([native.position[i] for i in range(3)]),
            native.bodyHeight,
            np.array([native.velocity[i] for i in range(3)]),
            native.yawSpeed,
            np.array([native.rangeObstacle[i] for i in range(8)]),
            [Cartesian.fromNative(native.footPosition2Body[i]) for i in range(4)],
            [Cartesian.fromNative(native.footSpeed2Body[i]) for i in range(4)],
            np.asarray(native.wirelessRemote, dtype=np.uint8),
            native.crc
        )

    def isValid(self):
        return (
            self.head[0] != 0 or
            self.head[1] != 0 or
            self.levelFlag != 0 or
            self.frameReserve != 0 or
            self.SN[0] != 0 or
            self.SN[1] != 0 or
            self.version[0] != 0 or
            self.version[1] != 0 or
            self.bandWidth != 0 or
            self.imu.isValid() or
            any([ms.isValid() for ms in self.motorState]) or
            self.bms.isValid() or
            np.any(self.footForce != 0) or
            np.any(self.footForceEst != 0) or
            self.mode != 0 or
            self.progress != 0 or
            self.gaitType != 0 or
            self.footRaiseHeight != 0 or
            np.any(self.position != 0) or
            self.bodyHeight != 0 or
            np.any(self.velocity != 0) or
            self.yawSpeed != 0 or
            np.any(self.rangeObstacle != 0) or
            any([cp.isValid() for cp in self.footPosition2Body]) or
            any([cp.isValid() for cp in self.footSpeed2Body]) or
            np.any(self.wirelessRemote != 0) or
            self.crc != 0
        )
    
class HighCmd(HasNativeRepr):
    def __init__(
        self,
        head: typing.Tuple[int, int],
        levelFlag: int,
        frameReserve: int,
        SN: typing.Tuple[int, int],
        version: typing.Tuple[int, int],
        bandWidth: int,
        mode: int, 
        # 0. idle, default stand
        # 1. force stand (controlled by dBodyHeight + ypr)
        # 2. target velocity walking (controlled by velocity + yawSpeed)
        # 3. target position walking (controlled by position + ypr[0])
        # 4. path mode walking (reserve for future release)
        # 5. position stand down. 
        # 6. position stand up 
        # 7. damping mode 
        # 8. recovery stand
        # 9. backflip
        # 10. jumpYaw
        # 11. straightHand
        # 12. dance1
        # 13. dance2
        gaitType:int, # 0.idle  1.trot  2.trot running  3.climb stair  4.trot obstacle
        speedLevel: int, # 0. default low speed. 1. medium speed 2. high speed. during walking, only respond MODE 3
        footRaiseHeight: float, # (unit: m, default: 0.08m), foot up height while walking, delta value
        bodyHeight: float, #(unit: m, default: 0.28m), delta value
        position: np.ndarray, #(unit: m), desired position in inertial frame
        euler: np.ndarray, #(unit: rad), roll pitch yaw in stand mode
        velocity: np.ndarray, #(unit: m/s), forwardSpeed, sideSpeed in body frame
        yawSpeed: float, #(unit: rad/s), rotateSpeed in body frame
        bms: BmsCmd,
        led: typing.List[LED],
        wirelessRemote: np.ndarray,
        crc: int        
    ):
        assert len(head) == 2
        assert len(SN) == 2
        assert len(version) == 2
        assert position.shape == (3,)
        assert euler.shape == (3,)
        assert velocity.shape == (2,)
        assert len(led) == 4
        assert wirelessRemote.shape == (40,)
        
        self.head = head
        self.levelFlag = levelFlag
        self.frameReserve = frameReserve
        self.SN = SN
        self.version = version
        self.bandWidth = bandWidth
        self.mode = mode
        self.gaitType = gaitType
        self.speedLevel = speedLevel
        self.footRaiseHeight = footRaiseHeight
        self.bodyHeight = bodyHeight
        self.position = position
        self.euler = euler
        self.velocity = velocity
        self.yawSpeed = yawSpeed
        self.bms = bms
        self.led = led
        self.wirelessRemote = wirelessRemote
        self.crc = crc
    
    def toNative(self):
        nat = robot_interface.HighCmd()
        return self.toNativeM(nat)
    
    def toNativeM(self, native):
        # assert len(self.head) == 2
        # assert len(self.SN) == 2
        # assert len(self.version) == 2
        # assert self.position.shape == (3,)
        # assert self.euler.shape == (3,)
        # assert self.velocity.shape == (2,)
        # assert len(self.led) == 4
        # assert self.wirelessRemote.shape == (40,)

        native.head[0] = self.head[0]
        native.head[1] = self.head[1]
        native.levelFlag = self.levelFlag
        native.frameReserve = self.frameReserve
        native.SN[0] = self.SN[0]
        native.SN[1] = self.SN[1]
        native.version[0] = self.version[0]
        native.version[1] = self.version[1]
        native.bandWidth = self.bandWidth
        native.mode = self.mode
        native.gaitType = self.gaitType
        native.speedLevel = self.speedLevel
        native.footRaiseHeight = self.footRaiseHeight
        native.bodyHeight = self.bodyHeight
        for i in range(3):
            native.position[i] = self.position[i]
        for i in range(3):
            native.euler[i] = self.euler[i]
        for i in range(3):
            native.velocity[i] = self.velocity[i]
        native.yawSpeed = self.yawSpeed
        #native.bms = self.bms.toNative()
        self.bms.toNativeM(native.bms)
        for i in range(4):
            #native.led[i] = self.led[i].toNative()
            self.led[i].toNativeM(native.led[i])
        for i in range(40):
            native.wirelessRemote[i] = self.wirelessRemote[i]
        native.crc = self.crc
    
        return native
    
    @staticmethod
    def fromNative(native):
        return __class__(
            (native.head[0], native.head[1]),
            native.levelFlag,
            native.frameReserve,
            (native.SN[0], native.SN[1]),
            (native.version[0], native.version[1]),
            native.bandWidth,
            native.mode,
            native.gaitType,
            native.speedLevel,
            native.footRaiseHeight,
            native.bodyHeight,
            np.array([native.position[i] for i in range(3)]),
            np.array([native.euler[i] for i in range(3)]),
            np.array([native.velocity[i] for i in range(3)]),
            native.yawSpeed,
            BmsCmd.fromNative(native.bms),
            [LED.fromNative(native.led[i]) for i in range(4)],
            np.asarray(native.wirelessRemote, dtype=np.uint8),
            native.crc
        )