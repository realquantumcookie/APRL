from rail_real_walker import *
from rail_real_walker.robots import Go1RealWalker
from unitree_go1_wrapper import XRockerBtnDataStruct
from rail_walker_interface import BaseWalker, BaseWalkerLocalizable, Walker3DVelocityEstimator
import numpy as np
import matplotlib.pyplot as plt

TICK_TIMESTEP = 0.03

if __name__ == "__main__":
    estimator_instance = DummyVelocityEstimator()
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
            lxly_arrow = plt.arrow(0, 0, 0, 0, head_width=0.05, width=0.05, color='r')
            rxry_arrow = plt.arrow(0, 0, 0, 0, head_width=0.05, width=0.05, color='b')
            plt.show()
            robot.reset()

            input("Press Enter to continue...")

            while True:
                robot.receive_observation()
                print(robot._last_state.wirelessRemote)
                wireless_remote = XRockerBtnDataStruct.fromNative(
                    robot._last_state.wirelessRemote
                )
                
                
                lxly_arrow.set_data(dx=wireless_remote.lx, dy=wireless_remote.ly)
                rxry_arrow.set_data(dx=wireless_remote.rx, dy=wireless_remote.ry)
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