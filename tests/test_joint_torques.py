from rail_real_walker import *
from rail_real_walker.robots import Go1RealWalker, Go1RealWalkerRemote
from rail_walker_interface import BaseWalker, BaseWalkerLocalizable, Walker3DVelocityEstimator
import numpy as np
import matplotlib.pyplot as plt

TICK_TIMESTEP = 0.03

def init_estimator() -> Walker3DVelocityEstimator:
    return DummyVelocityEstimator()

def interpolate_next_action(robot : Go1RealWalker, step : int):
    init_coefficient = (np.sin(np.pi * 2 * step / 50) + 1) / 2.0
    sit_coefficient = 1.0 - init_coefficient
    return init_coefficient * robot.joint_qpos_init + sit_coefficient * robot.joint_qpos_sitting


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
            FR_thigh_arrow = plt.arrow(0, 0, 0, 0, head_width=0.05, width=0.05, color='r')
            FL_thigh_arrow = plt.arrow(0, 0, 0, 0, head_width=0.05, width=0.05, color='b')
            plt.show()
            robot.reset()

            input("Press Enter to continue...")

            step = 0

            while True:
                robot.apply_action(interpolate_next_action(robot, step))
                robot.receive_observation()
                
                joint_torques = robot.get_joint_torques()
                joint_vels = robot.get_joint_qvel()
                FR_thigh_arrow.set_data(dx=joint_torques[1], dy=joint_vels[1])
                FL_thigh_arrow.set_data(dx=joint_torques[4], dy=joint_vels[4])
                
                plt.plot()
                plt.pause(TICK_TIMESTEP)
                step += 1
    except:
        import traceback; traceback.print_exc()
    
    finally:
        print("Closing...")
        robot.close()
        estimator_instance.close()
        plt.close()
        exit(0)