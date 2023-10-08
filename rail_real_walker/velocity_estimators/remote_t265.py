from rail_walker_interface import BaseWalker, Walker3DVelocityEstimatorLocal, Walker3DFrameQuatEstimator
from .remote_t265_runner import DataPack, RemoteT265Data
from typing import Optional, Any
import numpy as np
import transforms3d as tr3d
import time
import socket
from serde.msgpack import from_msgpack
import errno

class IntelRealsenseT265EstimatorRemote(Walker3DVelocityEstimatorLocal):
    def __init__(
        self,
        x_axis_on_robot: np.ndarray = np.array([1, 0, 0]),
        y_axis_on_robot: np.ndarray = np.array([0, 1, 0]),
        z_axis_on_robot: np.ndarray = np.array([0, 0, 1]),
        server_addr : tuple[str, int] = ("192.168.123.161",6000)
    ):
        Walker3DVelocityEstimatorLocal.__init__(self)
        Walker3DFrameQuatEstimator.__init__(self)
        self.socket_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket_client.setblocking(True)
        self.socket_client.settimeout(0.5)
        self.socket_client.connect(server_addr)
        self.socket_client.setblocking(False)
        
        self._server_addr = server_addr
        self.data_pack = DataPack(self.deal_with_data)

        self._enabled = True

        self._last_data : Optional[RemoteT265Data] = None
        self._last_data_t : float = 0.0

        self._local_velocity = np.zeros(3)
        self.setup_installation_quat_inv(x_axis_on_robot, y_axis_on_robot, z_axis_on_robot)
    
    def deal_with_data(self, command : bytes, data : bytes, is_last_data : bool, custom : Any) -> Any:
        if command == b"o" and is_last_data:
            try:
                new_data = from_msgpack(RemoteT265Data, data)
            except KeyboardInterrupt as e:
                raise e
            except BaseException as e:
                print("Error in remote T265 Data Decode", e)
                return

            new_t = time.time()
            if np.linalg.norm(new_data.rotation_wxyz) < 1e-6:
                return

            # if self._last_data_t > 0 and new_t - self._last_data_t < 1.0:
            #     dt = new_t - self._last_data_t
            #     tracker_velocity = (new_data.translation - self._last_data.translation) / dt
            #     tracker_local_velocity = tr3d.quaternions.rotate_vector(
            #         tracker_velocity,
            #         tr3d.quaternions.qinverse(self._last_data.rotation_wxyz)
            #     )
            # else:
            tracker_velocity = np.asarray(new_data.velocity)
            tracker_local_velocity = tr3d.quaternions.rotate_vector(
                tracker_velocity,
                tr3d.quaternions.qinverse(np.asarray(new_data.rotation_wxyz))
            )
            local_velocity = tr3d.quaternions.rotate_vector(
                tracker_local_velocity,
                self.installation_quat_inv
            )
            self._last_data = new_data
            self._last_data_t = new_t
            self._local_velocity = local_velocity

    def send_read_obs_data(self) -> None:
        pass
        # self.socket_client.send(self.data_pack.encode(b"o",b""))
    
    def try_receive_data(self) -> None:
        while True:
            try:
                data = self.socket_client.recv(8192)
                if not data:
                    raise RuntimeError("No data received from remote Go1 Runner! The TCP connection might be broken!")
                self.data_pack.feed_data(data, time.time())
                if len(data) < 8192:
                    return
            except socket.error as e:
                if e.errno == errno.EAGAIN or e.errno == errno.EWOULDBLOCK:
                    return
                else:
                    raise e

    @property
    def server_addr(self) -> tuple[str, int]:
        return self._server_addr

    @property
    def enabled(self) -> bool:
        return self._enabled
    
    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    def setup_installation_quat_inv(
        self, 
        x_axis_on_robot: np.ndarray,
        y_axis_on_robot: np.ndarray,
        z_axis_on_robot: np.ndarray
    ):
        assert np.any(x_axis_on_robot != 0) and np.any(y_axis_on_robot != 0) and np.any(z_axis_on_robot != 0)
        assert np.inner(x_axis_on_robot, y_axis_on_robot) == 0 and np.inner(x_axis_on_robot, z_axis_on_robot) == 0 and np.inner(y_axis_on_robot, z_axis_on_robot) == 0
        
        rotation_quat = tr3d.quaternions.mat2quat(
            np.hstack([
                x_axis_on_robot / np.linalg.norm(x_axis_on_robot), 
                y_axis_on_robot / np.linalg.norm(y_axis_on_robot), 
                z_axis_on_robot / np.linalg.norm(z_axis_on_robot),
            ])
        )
        self.installation_quat_inv = tr3d.quaternions.qinverse(rotation_quat)

    def _step(self, robot: BaseWalker, frame_quat: Optional[np.ndarray], dt: float):
        self.send_read_obs_data()
        self.try_receive_data()

    def __copy__(self):
        raise NotImplementedError()
    
    def __deepcopy__(self, memo):
        raise NotImplementedError()

    def _reset(self, robot: BaseWalker, frame_quat: Optional[np.ndarray]): 
        self.send_read_obs_data()
        self.try_receive_data()
        self._local_velocity = np.zeros(3)
    
    def get_3d_local_velocity(self) -> np.ndarray:
        return self._local_velocity
    
    def close(self) -> None:
        self.socket_client.close()
