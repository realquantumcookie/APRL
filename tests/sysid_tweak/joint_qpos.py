import unitree_go1_wrapper as go1_wrapper
import numpy as np
import time

print_idxes = list(range(3,6))

if __name__ == "__main__":
    udp = go1_wrapper.UDP(go1_wrapper.ControlLevel.LOW)
    low_cmd = udp.initCommunicationLowCmdData()
    udp.setToSendLow(low_cmd)
    udp.send()
    udp.recv()
    state = udp.getRecvLow()
    

    while True:
        udp.setToSendLow(low_cmd)
        udp.send()
        udp.recv()
        state = udp.getRecvLow()


        torques = np.array([
            state.motorState[i].tauEst for i in range(12)
        ])
        qpos = np.array([
            state.motorState[i].q for i in range(12)
        ])
        qvel = np.array([
            state.motorState[i].dq for i in range(12)
        ])
        print("Qpos", [qpos[i] for i in print_idxes])
    
