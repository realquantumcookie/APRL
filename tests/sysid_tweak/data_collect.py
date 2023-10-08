import unitree_go1_wrapper as go1_wrapper
import numpy as np
from response_curve import y_knee, y_abduction, y_hip #y_knee
import time

enabled_idx = [0,1,2]
#y_abduction = [go1_wrapper.GO1_HIP_INIT] * 5000
#y_hip = [go1_wrapper.GO1_THIGH_INIT] * 5000
#y_knee = [go1_wrapper.GO1_CALF_INIT] * 5000

if __name__ == "__main__":
    udp = go1_wrapper.UDP(go1_wrapper.ControlLevel.LOW)
    low_cmd = udp.initCommunicationLowCmdData()
    udp.setToSendLow(low_cmd)
    udp.send()
    udp.recv()
    state = udp.getRecvLow()
    
    torques_collected = []
    qpos_collected = []
    qvel_collected = []

    for i in range(12):
        low_cmd.motorCmd[i].Kp = 2.0
        low_cmd.motorCmd[i].Kd = 0.5
        low_cmd.motorCmd[i].tau = 0.0
        low_cmd.motorCmd[i].dq = 0.0
    for i in range(0, 12, 3):
        if i not in enabled_idx:
            low_cmd.motorCmd[i].q = 0.0
    for i in range(1, 12, 3):
        if i not in enabled_idx:
            low_cmd.motorCmd[i].q = 70.0/180.0*np.pi
    for i in range(2, 12, 3):
        if i not in enabled_idx:
            low_cmd.motorCmd[i].q = -150.0/180.0*np.pi

    for i in range(len(y_abduction)):
        target_abduction = y_abduction[i]
        target_hip = y_hip[i]
        target_knee = y_knee[i]
        print(target_abduction, target_hip, target_knee)
        for i in [0, 6]:
            if i in enabled_idx:
                low_cmd.motorCmd[i].q = target_abduction
        for i in [3, 9]:
            if i in enabled_idx:
                low_cmd.motorCmd[i].q = -target_abduction
        for i in range(1, 12, 3):
            if i in enabled_idx:
                low_cmd.motorCmd[i].q = target_hip
        for i in range(2, 12, 3):
            if i in enabled_idx:
                low_cmd.motorCmd[i].q = target_knee
        
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
        torques_collected.append(torques)
        qpos_collected.append(qpos)
        qvel_collected.append(qvel)
    np.savez_compressed(
        "data.npz",
        torques=torques_collected,
        qpos=qpos_collected,
        qvel=qvel_collected,
    )
