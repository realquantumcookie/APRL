import pyrealsense2 as rs
from rail_walker_interface import BaseWalker, Walker3DLocationEstimator, Walker3DFrameQuatEstimator
from typing import Optional
import multiprocessing
import numpy as np
import transforms3d as tr3d
import time

def normalize_rad(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi

"""
Intel Realsense T265 Velocity/Frame Quaternion Estimator
Please see axis definition here, you need to define how those axis are installed on your robot through the constructor

https://github.com/IntelRealSense/librealsense/blob/master/doc/t265.md
"""
class IntelRealsenseT265PositionEstimator(Walker3DLocationEstimator, Walker3DFrameQuatEstimator):
    def __init__(
        self,
        x_axis_on_robot: np.ndarray = np.array([1, 0, 0]),
        y_axis_on_robot: np.ndarray = np.array([0, 1, 0]),
        z_axis_on_robot: np.ndarray = np.array([0, 0, 1]),
    ):
        Walker3DLocationEstimator.__init__(self)
        Walker3DFrameQuatEstimator.__init__(self)
        self.comm_pipe : Optional[multiprocessing.Pipe] = None
        self.running_thread : Optional[multiprocessing.Process] = None
        self._enabled = True
        
        self._last_position = np.zeros(3)
        self._last_frame_quat = np.zeros(4)
        self._rs_quat_estimate = np.zeros(4)
        self.setup_installation_quat_inv(x_axis_on_robot, y_axis_on_robot, z_axis_on_robot)
        self.set_up_thread_runner()

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

    def __copy__(self):
        raise NotImplementedError()
    
    def __deepcopy__(self, memo):
        raise NotImplementedError()

    def update_position_estimate(self):
        local_position_rs_shift, rs_framequat = self.receive_position_shift_and_framequat()
        if not self.enabled:
            return

        rs_robot_framequat = tr3d.quaternions.qmult(
            self.installation_quat_inv,
            rs_framequat
        )
        self._rs_quat_estimate = rs_robot_framequat

        framequat_used = self._last_frame_quat if self._last_frame_quat is not None else rs_robot_framequat
        
        # quat_from_rs_to_robot_global = tr3d.quaternions.qmult(
        #     self.installation_quat_inv,
        #     framequat_used
        # )
        # global_position_shift = tr3d.quaternions.rotate_vector(
        #     local_position_rs_shift,
        #     quat_from_rs_to_robot_global
        # )
        
        local_position_shift = tr3d.quaternions.rotate_vector(
            local_position_rs_shift,
            self.installation_quat_inv
        )
        global_position_shift = tr3d.quaternions.rotate_vector(
            local_position_shift,
            framequat_used
        )
        self._last_position += global_position_shift

    def reset(self, robot: BaseWalker, frame_quat: Optional[np.ndarray]):
        if not np.all(self._last_frame_quat == 0): # If already initialized
            self.update_position_estimate()
        else:
            self.receive_position_shift_and_framequat() # Let RS thread clear up the accumulated position shift

        self._last_frame_quat = frame_quat
        Walker3DLocationEstimator.reset(
            self,
            robot,
            frame_quat
        )
    
    def step(self, robot: BaseWalker, frame_quat: Optional[np.ndarray], dt: float):
        self.update_position_estimate()
        self._last_frame_quat = frame_quat
        Walker3DLocationEstimator.step(
            self,
            robot,
            frame_quat,
            dt
        )
    
    def get_framequat_wijk(self) -> np.ndarray:
        return self._rs_quat_estimate
    
    def _estimate_location(self, robot: BaseWalker, frame_quat: Optional[np.ndarray]) -> np.ndarray:
        return self._last_position

    def receive_position_shift_and_framequat(self) -> tuple[np.ndarray, np.ndarray]:
        if self.comm_pipe is not None:
            self.comm_pipe.send(True)
            position_shift_local, framequat = self.comm_pipe.recv()
            return position_shift_local, framequat
        else:
            raise RuntimeError("Intel T265 thread is not running")
    
    def set_up_thread_runner(self):
        self.cleanup_thread_runner()
        self.comm_pipe, child_pipe = multiprocessing.Pipe()
        self.running_thread = multiprocessing.Process(
            target=__class__.thread_runer, 
            args=(child_pipe,),
            daemon=True
        )
        self.running_thread.start()

    def close(self):
        self.cleanup_thread_runner()

    def cleanup_thread_runner(self):
        if self.running_thread is not None:
            self.running_thread.terminate()
            self.running_thread = None
        
        if self.comm_pipe is not None:
            self.comm_pipe.close()
            self.comm_pipe = None

    @staticmethod
    def init_rs_stream():
        pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.pose)
        pipe.start(cfg)
        return pipe

    def thread_runer(
        comm_pipe: multiprocessing.Pipe
    ):
        print("Intel T265 thread started")
        try:
            rs_pipe = __class__.init_rs_stream()
        except:
            print("Failed to initialize realsense t265")
            comm_pipe.close()
            return
        print("Successfully initialized realsense t265")
        last_position = np.zeros(3)
        last_frame_quat = np.zeros(4)
        last_time = 0

        last_step_position = np.zeros(3)
        last_step_framequat = np.zeros(4)
        last_step_time = 0
        
        try:
            while True:
                t = time.time()

                if comm_pipe.poll():
                    comm_pipe.recv()
                    if last_step_time > 0 and last_time > 0:
                        step_dt = last_time - last_step_time
                        if step_dt > 0:
                            global_pos_shift = last_position - last_step_position
                        else:
                            global_pos_shift = np.zeros(3)
                        try:
                            if tr3d.quaternions.qnorm(last_step_framequat) < 1e-5:
                                print("Realsense t265 frame quat is zero, using zero pos shift")
                                local_pos_shift = np.zeros(3)
                            else:
                                local_pos_shift = tr3d.quaternions.rotate_vector(
                                    global_pos_shift,
                                    tr3d.quaternions.qinverse(last_step_framequat)
                                )
                        except:
                            print("Error in realsense t265 tr3d.quaternions.rotate_vector()")
                            local_pos_shift = np.zeros(3)
                    else:
                        local_pos_shift = np.zeros(3)
                    
                    comm_pipe.send((local_pos_shift, last_frame_quat))
                    last_step_position = last_position
                    last_step_framequat = last_frame_quat
                    last_step_time = t
                
                try:
                    frames = rs_pipe.wait_for_frames(20) # 20 ms timeout
                    pose = frames.get_pose_frame()
                    if pose:
                        data = pose.get_pose_data()
                        last_position = np.array([data.translation.x, data.translation.y, data.translation.z])
                        last_frame_quat = np.array([data.rotation.w, data.rotation.x, data.rotation.y, data.rotation.z])
                        last_time = t
                except:
                    print("Error in realsense t265 rs_pipe.wait_for_frames()")


        finally:
            rs_pipe.stop()
            comm_pipe.close()