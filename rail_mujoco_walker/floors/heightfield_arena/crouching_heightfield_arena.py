from .heightfield_arena import HeightFieldArena
import numpy as np

CROUCHING_LOCATION = np.array([0.05, 0.18, 1.01])
CROUCHING_SIZE = np.array([0.03, 0.05, 0.5])
CROUCHING_YAW = 10.0/180 * np.pi

def _helper_is_in_croching_range(pos_2d : np.ndarray, crouching_location_2d : np.ndarray, crouching_size_2d : np.ndarray, crouching_yaw : float):
    assert pos_2d.shape == (2,) and crouching_location_2d.shape == (2,) and crouching_size_2d.shape == (2,)
    delta_pos = pos_2d - crouching_location_2d
    x_axis_unit = [np.cos(crouching_yaw), np.sin(crouching_yaw)]
    y_axis_unit = [-np.sin(crouching_yaw), np.cos(crouching_yaw)]
    x_scalar = np.inner(delta_pos, x_axis_unit)
    y_scalar = np.inner(delta_pos, y_axis_unit)
    # https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-geom-size
    # it is safe to do so because mujoco has size specified by half size
    return np.abs(x_scalar) < crouching_size_2d[0] and np.abs(y_scalar) < crouching_size_2d[1]


class CrouchingHeightfieldArena(
    HeightFieldArena
):
    def __init__(self, scale, name="crouching_heightfield_arena"):
        super().__init__(scale, name=name)
    
    def _build(self, name):
        super()._build(name)

        size = [self.scale * s for s in CROUCHING_SIZE[:2]] + [CROUCHING_SIZE[2]]
        pos = [self.scale * p for p in CROUCHING_LOCATION[:2]] + [CROUCHING_LOCATION[2]]
        euler = (0.0, 0.0, CROUCHING_YAW)
        
        self._crouch_overhead = self._mjcf_root.worldbody.add(
            'geom',
            name='crouch_overhead',
            type='box',
            rgba=(1.0, 1.0, 1.0, 0.3),
            pos=pos,
            size=size,
            euler=euler
        )

    def should_crouch(self, robot_pos : np.ndarray):
        assert robot_pos.ndim == 1 and robot_pos.shape[0] >= 2
        return _helper_is_in_croching_range(
            robot_pos[:2],
            CROUCHING_LOCATION[:2] * self.scale,
            CROUCHING_SIZE[:2] * self.scale + 1.0, # 1.0 to account for the robot's size
            CROUCHING_YAW
        )

    @staticmethod
    def get_should_crouch_callback(scale : float):
        def is_in_crouching_range(
            robot_pos : np.ndarray
        ) -> bool:
            assert robot_pos.ndim == 1 and robot_pos.shape[0] >= 2
            return _helper_is_in_croching_range(
                robot_pos[:2],
                CROUCHING_LOCATION[:2] * scale,
                CROUCHING_SIZE[:2] * scale + 0.5, # 0.5 to account for the robot's size
                CROUCHING_YAW
            )
        return is_in_crouching_range