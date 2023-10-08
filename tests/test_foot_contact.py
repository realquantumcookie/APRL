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
        Kp=30,
        Kd=3
    )
    
    try:
        print("Press Enter to continue")
        input()
        robot.reset()

        while True:
            robot.apply_action(robot.joint_qpos_init)
            robot.receive_observation()
            foot_contact = robot.get_foot_force()
            print("Foot Contact: ", foot_contact)
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        import traceback
        traceback.print_exc()
    finally:
        robot.close()