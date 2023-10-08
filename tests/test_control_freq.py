from rail_real_walker import *
from rail_real_walker.robots import Go1RealWalkerRemote
from rail_walker_interface import BaseWalker, BaseWalkerLocalizable, Walker3DVelocityEstimator
import numpy as np
import tqdm

TICK_TIMESTEP = 0.03


if __name__ == "__main__":
    estimator_instance = DummyVelocityEstimator()
    robot = Go1RealWalkerRemote(
        estimator_instance,
        power_protect_factor=0.5,
        Kp=40,
        Kd=5
    )
    
    try:
        print("Press Enter to continue")
        input()
        robot.reset()

        for _ in tqdm.trange(0, 100000):
            robot.apply_action(robot.joint_qpos_init)
            robot.receive_observation()
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        import traceback
        traceback.print_exc()
    finally:
        robot.close()