# https://github.com/unitreerobotics/unitree_legged_sdk/blob/go1/include/unitree_legged_sdk/joystick.h
from .comm import HasNativeRepr, State
from .native_interface import robot_interface
import numpy as np
from typing import Tuple

class XKeySwitchUnionComponents(HasNativeRepr):
    def __init__(
        self,
        R1: int,
        L1: int,
        start: int,
        select: int,
        R2: int,
        L2: int, 
        F1: int,
        F2: int,
        A: int,
        B: int,
        X: int,
        Y: int,
        up: int,
        right: int,
        down: int,
        left: int
    ):
        self.R1 = R1
        self.L1 = L1
        self.start = start
        self.select = select
        self.R2 = R2
        self.L2 = L2
        self.F1 = F1
        self.F2 = F2
        self.A = A
        self.B = B
        self.X = X
        self.Y = Y
        self.up = up
        self.right = right
        self.down = down
        self.left = left
    
    def toNative(self):
        native = np.zeros(16, dtype=np.uint8)
        return self.toNativeM(native)

    def toNativeM(self, native : np.ndarray):
        assert native.shape == (16,) and native.dtype == np.uint8
        native[0] = self.R1
        native[1] = self.L1
        native[2] = self.start
        native[3] = self.select
        native[4] = self.R2
        native[5] = self.L2
        native[6] = self.F1
        native[7] = self.F2
        native[8] = self.A
        native[9] = self.B
        native[10] = self.X
        native[11] = self.Y
        native[12] = self.up
        native[13] = self.right
        native[14] = self.down
        native[15] = self.left
    
    @staticmethod
    def fromNative(native : np.ndarray):
        # 16 bytes uint8 array
        assert native.shape == (16,) and native.dtype == np.uint8
        return __class__(
            *native
        )




class XKeySwitchUnion(HasNativeRepr):
    def __init__(
        self,
        value : int,
        components : XKeySwitchUnionComponents
    ):
        # It's a union type in C++ but we can't do that in Python
        # So we'll just have to keep track of the value and the components
        self.value = value
        self.components = components
    
    def toNative(self):
        return self.components.toNative()
    
    def toNativeM(self, native):
        return self.components.toNativeM(native)

    @staticmethod
    def fromNative(native : np.ndarray):
        assert native.shape == (16,) and native.dtype == np.uint8
        return __class__(
            native[:2].view(np.uint16)[0],
            XKeySwitchUnionComponents.fromNative(native)
        )

class XRockerBtnDataStruct(HasNativeRepr):
    def __init__(
        self,
        head: Tuple[int, int],
        btn: XKeySwitchUnion,
        lx: float,
        ly: float,
        rx: float,
        ry: float,
        L2: float # not sure what this is for?
    ):
        assert len(head) == 2
        self.head = head
        self.btn = btn
        self.lx = lx
        self.ly = ly
        self.rx = rx
        self.ry = ry
        self.L2 = L2
    
    def toNative(self) -> np.ndarray:
        native = np.zeros(40, dtype=np.uint8)
        return self.toNativeM(native)

    def toNativeM(self, native : np.ndarray):
        assert native.shape == (40,) and native.dtype == np.uint8
        native[:2] = np.array(self.head, dtype=np.uint8)
        self.btn.toNativeM(native[2:18])
        native[4:4+4*5] = np.array([self.lx, self.rx, self.ry, self.L2, self.ly], dtype=np.float16).view(np.uint8)
        return native

    @staticmethod
    def fromNative(native : np.ndarray):
        assert native.shape == (40,) and native.dtype == np.uint8
        head = tuple(native[:2]) #byte aligned
        
        float_values = native[4:4 + 4*5].view(np.float32)
        lx, rx, ry, L2, ly = float_values

        btn = XKeySwitchUnion.fromNative(native[2:18]) # Doesn't work for now
        return __class__(
            head,
            btn,
            lx,
            ly,
            rx,
            ry,
            L2
        )

    @staticmethod
    def max_mag():
        return np.sqrt(2)