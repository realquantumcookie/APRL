from rail_real_walker import *
from rail_real_walker.robots import Go1RealWalker, Go1RealWalkerRemote
from rail_walker_interface import BaseWalker, BaseWalkerLocalizable, Walker3DVelocityEstimator
import numpy as np
import matplotlib.pyplot as plt

TICK_TIMESTEP = 0.03

def init_estimator() -> Walker3DVelocityEstimator:
    return IntelRealsenseT265Estimator(
        x_axis_on_robot=np.array([-1, 0, 0]),
        y_axis_on_robot=np.array([0, 0, 1]),
        z_axis_on_robot=np.array([0, 1, 0]),
    )
    # return Go1ForwardKinematicsVelocityProvider()
    # return KalmanFilterFusedVelocityEstimator()

if __name__ == "__main__":
    estimator_instance = init_estimator()
    print("Estimator type: ", type(estimator_instance))
    robot = Go1RealWalker(
        estimator_instance,
        power_protect_factor=0.5,
        Kp=40,
        Kd=5
    )
    
    try:
        with plt.ion():
            plt.xlim(-1, 1)
            plt.ylim(-1, 1)
            local_arrow = plt.arrow(0, 0, 0, 0, head_width=0.05, width=0.05, color='r')
            global_arrow = plt.arrow(0, 0, 0, 0, head_width=0.05, width=0.05, color='b')
            plt.show()
            robot.reset()

            input("Press Enter to continue...")

            while True:
                robot.apply_action(robot.joint_qpos_init)
                robot.receive_observation()
                
                local_velocity = robot.get_3d_local_velocity()
                local_arrow.set_data(dx=local_velocity[0], dy=local_velocity[1])
                
                global_velocity = robot.get_3d_linear_velocity()
                global_arrow.set_data(dx=global_velocity[0], dy=global_velocity[1])
                plt.plot()
                plt.pause(TICK_TIMESTEP)
    except:
        import traceback; traceback.print_exc()
    
    finally:
        print("Closing...")
        robot.close()
        estimator_instance.close()
        plt.close()
        exit(0)