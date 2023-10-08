from unitree_go1_wrapper.native_interface import robot_interface
import unitree_go1_wrapper as go1_wrapper
import numpy as np
from response_curve import y_knee #y_abduction, y_hip, y_knee
import time

y_abduction = [go1_wrapper.GO1_HIP_INIT] * 5000
y_hip = [go1_wrapper.GO1_THIGH_INIT] * 5000
#y_knee = [go1_wrapper.GO1_CALF_INIT] * 5000

if __name__ == "__main__":
    udp = robot_interface.UDP(go1_wrapper.ControlLevel.LOW.value, 8080, "192.168.123.10", 8007)
    low_cmd = robot_interface.LowCmd()
    state = robot_interface.LowState()
    udp.InitCmdData(low_cmd)

    udp.SetSend(low_cmd)
    udp.Send
    udp.Recv()
    udp.GetRecv(state)
    
    torques_collected = []
    qpos_collected = []
    qvel_collected = []

    # low_cmd.motorCmd[0].Kp = 5.0
    # low_cmd.motorCmd[1].Kp = 5.0
    low_cmd.motorCmd[5].Kp = 60.0
    # low_cmd.motorCmd[0].Kd = 2.0
    # low_cmd.motorCmd[1].Kd = 2.0
    low_cmd.motorCmd[5].Kd = 50.0
    for i in range(len(y_abduction)):
        udp.Recv()
        udp.GetRecv(state)

        target_abduction = y_abduction[i]
        target_hip = y_hip[i]
        target_knee = y_knee[i]
        print(target_abduction, target_hip, target_knee)
        # low_cmd.motorCmd[0].q = target_abduction
        # low_cmd.motorCmd[1].q = target_hip
        low_cmd.motorCmd[5].q = target_knee
        low_cmd.motorCmd[5].dq = 0.0
        low_cmd.motorCmd[5].tau = 0.0
        low_cmd.motorCmd[5].Kp = 60.0
        low_cmd.motorCmd[5].Kd = 50.0
        udp.SetSend(low_cmd)
        udp.Send()
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
        time.sleep(0.002)
    np.savez_compressed(
        "data.npz",
        torques=torques_collected,
        qpos=qpos_collected,
        qvel=qvel_collected,
    )
