from rail_real_walker import *
from rail_real_walker.robots import Go1RealWalker
from rail_walker_interface import BaseWalker, BaseWalkerLocalizable, Walker3DVelocityEstimator
import numpy as np
import matplotlib.pyplot as plt

TICK_TIMESTEP = 0.03


if __name__ == "__main__":
    estimator_instance = DummyVelocityEstimator()
    robot = Go1RealWalker(
        estimator_instance,
        power_protect_factor=0.5,
        Kp=40,
        Kd=3
    )
    delta_pos = np.zeros(3)
    
    try:
        print("Press Enter to continue")
        input()
        robot.reset()

        while True:
            robot.receive_observation()
            ipt = input("Input delta pos: ")
            delta_pos = np.array([float(x) for x in ipt.split(",")][:3])
            delta_pos_left = delta_pos.copy()
            delta_pos_left[0] = -delta_pos_left[0]
            delta_pos_real = np.asarray([delta_pos, delta_pos_left, delta_pos, delta_pos_left]).reshape((12,))
            target_pos = robot.joint_qpos_init + delta_pos_real
            print(delta_pos, delta_pos_real, target_pos)
            robot.apply_action(target_pos)
            
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        import traceback
        traceback.print_exc()
    finally:
        robot.close()