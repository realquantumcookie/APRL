from dm_control.rl import control
import mujoco
import numpy as np
from dm_control.mjcf.physics import Physics
from typing import Optional


_MAX_SETTLE_PHYSICS_ATTEMPTS = 1
_MIN_SETTLE_PHYSICS_TIME = 0.0
_MAX_SETTLE_PHYSICS_TIME = 2.0

_SETTLE_QVEL_TOL = 1e-3

def settle_physics(
    physics: Physics,
    robot,
    qvel_tol=_SETTLE_QVEL_TOL,
    max_attempts=_MAX_SETTLE_PHYSICS_ATTEMPTS,
    min_time=_MIN_SETTLE_PHYSICS_TIME,
    max_time=_MAX_SETTLE_PHYSICS_TIME,
) -> bool:
    """Steps the physics until the robot root body is at rest."""
    sensor_binding = physics.bind(robot.root_body_linvel_sensor)
    for _ in range(max_attempts):
        original_time = physics.data.time
        while physics.data.time - original_time < max_time:
            physics.step()
            max_qvel = np.max(np.abs(sensor_binding.sensordata))
            if (max_qvel < qvel_tol) and (physics.data.time - original_time) > min_time:
                return True
        physics.data.time = original_time
    return False

def find_dm_control_non_contacting_height(
    physics : Physics,
    walker,
    x_pos : float = 0.0,
    y_pos : float = 0.0,
    qpos : Optional[np.ndarray]=None,
    quat : Optional[np.ndarray]=None,
    maxiter : int=1000
):
    z_pos = 0.0  # Start embedded in the floor.
    num_contacts = 1
    count = 1
    # Move up in 1cm increments until no contacts.
    while num_contacts > 0:
        try:
            with physics.reset_context():
                walker.set_pose(physics, [x_pos, y_pos, z_pos], quaternion=quat)
                if qpos is not None:
                    physics.bind(walker.joints).qpos[:] = qpos.copy()
                
        except control.PhysicsError:
            # We may encounter a PhysicsError here due to filling the contact
            # buffer, in which case we simply increment the height and continue.
            pass
        num_contacts = physics.data.ncon

        z_pos += 0.01
        count += 1
        if count > maxiter:
            raise ValueError('maxiter reached: possibly contacts in null pose of body.')

def find_dm_control_non_contacting_qpos(
    physics : Physics, 
    random_state : np.random.RandomState, 
    walker, 
    maxiter : int =1000
):
    num_contacts = 1
    count = 1
    # Move up in 1cm increments until no contacts.
    while num_contacts > 0:
        try:
            with physics.reset_context():
                walker.set_pose(physics, [0, 0, 100])
                joints_range = physics.bind(walker.joints).range
                qpos = random_state.uniform(joints_range[:, 0],
                                            joints_range[:, 1])
                walker.configure_joints(physics, qpos)
        except control.PhysicsError:
            # We may encounter a PhysicsError here due to filling the contact
            # buffer, in which case we simply increment the height and continue.
            pass
        num_contacts = physics.data.ncon

        count += 1

        if count > maxiter:
            raise ValueError(
                'maxiter reached: possibly contacts in null pose of body.')

    walker.set_pose(physics, [0, 0, 0])

    if num_contacts == 0:
        return qpos
    else:
        return None
