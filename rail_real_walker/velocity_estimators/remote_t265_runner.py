import socket
from serde import serde
from serde.msgpack import to_msgpack
from dataclasses import dataclass
import time
import errno
import pyrealsense2 as rs
import numpy as np
from enum import Enum
import select
from typing import Any
import os
import psutil
from typing import Tuple

class DataPack:
    def __init__(self, deal_command_callback):
        self.remaining_data = b""
        self.deal_command_callback = deal_command_callback
    
    def feed_data(self, dat: bytes, custom : Any) -> Any:
        self.remaining_data += dat
        return self.stream_decode(custom)
    
    def clear_data(self) -> None:
        self.remaining_data = b""

    def stream_decode(self, custom : Any) -> Any:
        decoded = True
        ret = None
        while decoded:
            decoded, ret = self.try_decode(custom)
        return ret

    def try_decode(self, custom: Any) -> Tuple[bool, Any]:
        if len(self.remaining_data) < 3:
            return False, None
        command = self.remaining_data[0:1]
        length = int.from_bytes(self.remaining_data[1:3], 'big')

        if len(self.remaining_data) < length + 3:
            return False, None
        
        if len(self.remaining_data) < length + 3 + 3:
            last_data = True
        else:
            next_length = int.from_bytes(self.remaining_data[length+3+1:length+3+3], 'big')
            last_data = not len(self.remaining_data) >= (length+3)+3+next_length

        data = self.remaining_data[3:length + 3]
        ret = self.deal_command_callback(command, data, last_data, custom)
        self.remaining_data = self.remaining_data[length + 3:]
        return True, ret
    
    @staticmethod
    def encode(command : bytes, data : bytes) -> bytes:
        length = len(data)
        enc = command + length.to_bytes(2, 'big') + data
        return enc

class RemoteT265Confidence(Enum):
    FAILED = 0x0
    LOW = 0x1
    MEDIUM = 0x2
    HIGH = 0x3

@serde
@dataclass
class RemoteT265Data:
    # acceleration: np.ndarray
    # angular_acceleration: np.ndarray
    # angular_velocity: np.ndarray
    # mapper_confidence: int
    tracker_confidence: int
    rotation_wxyz: Tuple[float, float, float, float]
    translation: Tuple[float, float, float]
    velocity: Tuple[float, float, float]

def empty_t265_data() -> RemoteT265Data:
    return RemoteT265Data(
        # acceleration=np.zeros(3),
        # angular_acceleration=np.zeros(3),
        # angular_velocity=np.zeros(3),
        # mapper_confidence=RemoteT265Confidence.FAILED.value,
        tracker_confidence=RemoteT265Confidence.FAILED.value,
        rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
        translation=(0.0, 0.0, 0.0),
        velocity=(0.0, 0.0, 0.0)
    )

class RemoteT265Updater:
    def __init__(
        self,
        wait_ms : int
    ):
        self.wait_ms = wait_ms
        self.last_data : RemoteT265Data = empty_t265_data()
        self.last_time = 0
        self.rs_pipe = None

    @staticmethod
    def create_rs_stream():
        pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.pose)
        pipe.start(cfg)
        return pipe

    def try_init_rs(self) -> bool:
        if self.rs_pipe is None:
            print("Initializing realsense t265")
            try:
                self.rs_pipe = __class__.create_rs_stream()
            except RuntimeError as e:
                return False
        return True
    
    @property
    def is_rs_inited(self) -> bool:
        return self.rs_pipe is not None
    
    @staticmethod
    def convert_rs_pose_to_t265_data(pose : rs.pose) -> RemoteT265Data:
        return RemoteT265Data(
            # acceleration=np.array([pose.acceleration.x, pose.acceleration.y, pose.acceleration.z]),
            # angular_acceleration=np.array([pose.angular_acceleration.x, pose.angular_acceleration.y, pose.angular_acceleration.z]),
            # angular_velocity=np.array([pose.angular_velocity.x, pose.angular_velocity.y, pose.angular_velocity.z]),
            # mapper_confidence=int(pose.mapper_confidence),
            tracker_confidence=int(pose.tracker_confidence),
            rotation_wxyz=(pose.rotation.w, pose.rotation.x, pose.rotation.y, pose.rotation.z),
            translation=(pose.translation.x, pose.translation.y, pose.translation.z),
            velocity=(pose.velocity.x, pose.velocity.y, pose.velocity.z)
        )

    def try_update(self) -> bool:
        if self.rs_pipe is None:
            return False
        try:
            frames = self.rs_pipe.wait_for_frames(self.wait_ms) # variable timeout
        except RuntimeError:
            return False
        t = time.time()
        pose = frames.get_pose_frame()
        if pose:
            data = pose.get_pose_data()
            self.last_data = __class__.convert_rs_pose_to_t265_data(data)
            self.last_time = t
            return True
        return False

    def close(self) -> None:
        if self.rs_pipe is not None:
            self.rs_pipe.stop()
            self.rs_pipe = None

    def __del__(self):
        self.close()


class RemoteT265VelocityEstimatorRunner:
    def __init__(
        self,
        bind_address : Tuple[str, int] = ("", 6000),
        wait_ms : int = 20
    ):
        self.tcp_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_server.bind(bind_address)
        self.tcp_server.listen(1)
        self.tcp_server.setblocking(False)
        self.tcp_clients : list[socket.socket] = []

        self._bind_address = bind_address
        self.updater = RemoteT265Updater(wait_ms=wait_ms)
        self.data_pack = DataPack(self.deal_command)
        self._next_to_send = self.data_pack.encode(
            b"o",
            to_msgpack(empty_t265_data())
        )
        self._updater_fail_count : int = 0
    
    def deal_command(self, command : bytes, data : bytes, is_last_data : bool, custom : socket.socket) -> Any:
        if command == b"o":
            custom.send(self._next_to_send)

    @property
    def bind_address(self) -> Tuple[str, int]:
        return self._bind_address

    def broadcast_data(self, data : bytes) -> None:
        for client in self.tcp_clients:
            try:
                num_sent = client.send(data)
            except:
                self.disconnect(client)
                continue
            if num_sent <= 0:
                self.disconnect(client)

    def disconnect(self, socket : socket.socket) -> None:
        print("Disconnecting Socket", socket)
        try:
            self.tcp_clients.remove(socket)
        except ValueError:
            pass
        socket.close()
        if len(self.tcp_clients) <= 0:
            self.data_pack.clear_data()

    def init_rs_until_success(self) -> None:
        while not self.updater.try_init_rs():
            time.sleep(5.0)
        print("Realsense t265 initialized")

    def try_update_t265(self) -> None:
        if self._updater_fail_count > 50:
            print("Reinitializing realsense t265 due to too many failures")
            self.init_rs_until_success()
            self._updater_fail_count = 0
        if self.updater.try_update():
            self._updater_fail_count = 0
            if len(self.tcp_clients) > 0:
                to_send = self.data_pack.encode(
                    b"o",
                    to_msgpack(self.updater.last_data)
                )
                self._next_to_send = to_send
                self.broadcast_data(to_send)
        else:
            self._updater_fail_count += 1

    def run(self):
        pid = os.getpid()
        process = psutil.Process(pid)
        try:
            process.nice(-5)
        except psutil.AccessDenied:
            print("Cannot set process priority, please run as root!")

        self.init_rs_until_success()
        try:
            while True:
                self.try_update_t265()
                if len(self.tcp_clients) <= 0:
                    time.sleep(0.1)
                
                readable, writable, errored = select.select([self.tcp_server] + self.tcp_clients, [], [], 0.0)
                for s in readable:
                    if s is self.tcp_server:
                        client_socket, address = self.tcp_server.accept()
                        self.tcp_clients.append(client_socket)
                        print ("Connection from", address)
                    else:
                        try:
                            data = s.recv(1024)
                        except socket.error as e:
                            if e.errno == errno.ECONNRESET:
                                self.disconnect(s)
                                break
                            elif e.errno != errno.EAGAIN and e.errno != errno.EWOULDBLOCK:
                                raise e
                            continue
                        if data:
                            self.data_pack.feed_data(data, s)
                        else:
                            self.disconnect(s)
                            break
        finally:
            process.nice(5)
    
    def close(self):
        self.updater.close()

    def __del__(self):
        self.close()
