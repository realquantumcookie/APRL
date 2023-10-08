from rail_walker_interface import BaseWalker
import numpy as np
import typing

def calculate_energy(joint_qvel : np.ndarray, joint_torques : np.ndarray) -> float:
    return np.sum(np.abs(joint_qvel * joint_torques))

def calculate_qvel_penalty(robot: BaseWalker) -> typing.Tuple[float, float]:
    qvel = robot.get_joint_qvel()
    qvel_penalty = -np.exp(-0.1*np.linalg.norm(qvel))

    return qvel_penalty, qvel

def near_quadratic_bound(value, target, left_margin, right_margin, out_of_margin_activation : str | None = "linear", power = 2.0, value_at_margin = 0.0):
    delta = value-target
    fract = delta/right_margin if delta > 0 else delta/left_margin
    
    if out_of_margin_activation is None or out_of_margin_activation != "near_quadratic":
        clipped_fract = np.clip(fract, -1.0, 1.0)
        rew = 1 - (1-value_at_margin) * (np.abs(clipped_fract) ** power)
        oodfract = fract - clipped_fract
        if out_of_margin_activation == "linear":
            rew -= (1-value_at_margin) * np.abs(oodfract)
        elif out_of_margin_activation == "quadratic":
            rew -= (1-value_at_margin) * (oodfract ** 2)
        elif out_of_margin_activation == "gaussian":
            rew += value_at_margin * np.exp(-oodfract**2/0.25)
    elif out_of_margin_activation == "near_quadratic":
        rew = 1 - (1-value_at_margin) * (np.abs(fract) ** power)
    return rew

def calculate_torque(current_qpos : np.ndarray, current_qvel : np.ndarray, target_qpos : np.ndarray, Kp : float, Kd : float):
    return Kp * (target_qpos - current_qpos) - Kd * current_qvel

def calculate_gaussian_activation(x : float | np.ndarray):
    return np.exp(-np.linalg.norm(x)**2/0.25)
