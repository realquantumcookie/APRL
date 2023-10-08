import pyrealsense2 as rs
from rail_walker_interface import BaseWalker, Walker3DVelocityEstimatorLocal, Walker3DFrameQuatEstimator
from typing import Optional
import multiprocessing
import numpy as np
import transforms3d as tr3d
import time

"""
Intel Realsense T265 Velocity/Frame Quaternion Estimator
Please see axis definition here, you need to define how those axis are installed on your robot through the constructor

https://github.com/IntelRealSense/librealsense/blob/master/doc/t265.md
"""
class IntelRealsenseT265Estimator(Walker3DVelocityEstimatorLocal, Walker3DFrameQuatEstimator):
    def __init__(
        self,
        x_axis_on_robot: np.ndarray = np.array([1, 0, 0]),
        y_axis_on_robot: np.ndarray = np.array([0, 1, 0]),
        z_axis_on_robot: np.ndarray = np.array([0, 0, 1]),
    ):
        Walker3DVelocityEstimatorLocal.__init__(self)
        Walker3DFrameQuatEstimator.__init__(self)
        self.comm_pipe : Optional[multiprocessing.Pipe] = None
        self.running_thread : Optional[multiprocessing.Process] = None
        self._enabled = True

        self._local_velocity = np.zeros(3)
        self._last_frame_quat = np.zeros(4)
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

    def step(self, robot: BaseWalker, frame_quat: Optional[np.ndarray], dt: float):
        self._step(robot, frame_quat, dt)
        local_velocity = self.get_3d_local_velocity()
        self._global_velocity = tr3d.quaternions.rotate_vector(local_velocity, self._last_frame_quat)

    def __copy__(self):
        raise NotImplementedError()
    
    def __deepcopy__(self, memo):
        raise NotImplementedError()

    def _reset(self, robot: BaseWalker, frame_quat: Optional[np.ndarray]): 
        self._local_velocity = np.zeros(3)
        self._last_frame_quat = np.zeros(4)
        self.receive_position_and_framequat()

    def receive_position_and_framequat(self) -> tuple[np.ndarray, np.ndarray]:
        if self.comm_pipe is not None:
            self.comm_pipe.send(True)
            position, framequat = self.comm_pipe.recv()
            return position, framequat
        else:
            print("Realsense t265 not initialized")
            return np.zeros(3), np.zeros(4)

    def _step(self, robot: BaseWalker, frame_quat: Optional[np.ndarray], dt: float):
        tracker_velocity, received_quat = self.receive_position_and_framequat()
        if np.any(np.isnan(tracker_velocity)):
            print("Nan in realsense t265 step")
        
        tracker_velocity = np.nan_to_num(tracker_velocity, nan=0.0)
        #print(tracker_velocity,received_quat)
        if self.enabled:
            self._local_velocity = tr3d.quaternions.rotate_vector(
                tracker_velocity,
                self.installation_quat_inv
            )
        else:
            self._local_velocity = np.zeros(3)
        
        if frame_quat is None:
            #print(self.installation_quat_inv, received_quat)
            self._last_frame_quat = tr3d.quaternions.qmult(self.installation_quat_inv, received_quat)
        else:
            self._last_frame_quat = frame_quat
    
    def get_3d_local_velocity(self) -> np.ndarray:
        return self._local_velocity

    def get_framequat_wijk(self) -> np.ndarray:
        return self._last_frame_quat
    
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
        last_step_velocity = np.zeros(3)
        
        try:
            while True:
                t = time.time()

                if comm_pipe.poll():
                    comm_pipe.recv()
                    if last_step_time > 0 and last_time > 0:
                        step_dt = last_time - last_step_time
                        if step_dt > 0:
                            global_velocity = (last_position - last_step_position) / step_dt
                            try:
                                if tr3d.quaternions.qnorm(last_step_framequat) < 1e-5:
                                    print("Realsense t265 frame quat is zero, using zero velocity")
                                    local_velocity = np.zeros(3)
                                else:
                                    local_velocity = tr3d.quaternions.rotate_vector(
                                        global_velocity,
                                        tr3d.quaternions.qinverse(last_step_framequat)
                                    )
                            except:
                                print("Error in realsense t265 tr3d.quaternions.rotate_vector()")
                                local_velocity = np.zeros(3)
                        else:
                            global_velocity = np.zeros(3)
                            local_velocity = last_step_velocity
                        
                        if np.any(np.isnan(local_velocity)):
                            print("Nan in realsense t265 local_velocity")
                            print(global_velocity, last_step_framequat)
                            local_velocity = last_step_velocity

                    else:
                        local_velocity = np.zeros(3)
                    
                    local_velocity = np.clip(local_velocity, -1.5, 1.5)
                    comm_pipe.send((local_velocity, last_frame_quat))
                    last_step_position = last_position
                    last_step_framequat = last_frame_quat
                    last_step_time = t
                    last_step_velocity = local_velocity
                
                try:
                    frames = rs_pipe.wait_for_frames(20) # 20 ms timeout
                    pose = frames.get_pose_frame()
                    if pose:
                        data = pose.get_pose_data()
                        new_position = np.array([data.translation.x, data.translation.y, data.translation.z])
                        new_frame_quat = np.array([data.rotation.w, data.rotation.x, data.rotation.y, data.rotation.z])
                        if tr3d.quaternions.qnorm(new_frame_quat) < 1e-5:
                            print("Realsense t265 frame quat is zero!")
                            continue
                        last_position = new_position
                        last_frame_quat = new_frame_quat
                        last_time = t
                except RuntimeError:
                    print("Error in realsense t265 rs_pipe.wait_for_frames()")
                except KeyboardInterrupt:
                    print("Keyboard interrupt in realsense t265 thread")
                    import traceback; traceback.print_exc()
                    break

        finally:
            rs_pipe.stop()
            comm_pipe.close()
            print("Intel T265 thread ended")