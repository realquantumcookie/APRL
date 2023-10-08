import mujoco
import numpy as np

def add_arrow_to_mjv_scene(scene : mujoco.MjvScene, point1 : np.ndarray, point2 : np.ndarray, radius : float, rgba : np.ndarray):
    if scene.ngeom >= scene.maxgeom:
        return

    scene.ngeom += 1  # increment ngeom

    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom-1], # geom
        mujoco.mjtGeom.mjGEOM_ARROW, # type
        np.zeros(3), # size
        np.zeros(3), # pos
        np.zeros(9), # mat
        rgba.astype(np.float32), # mat
    )
    
    mujoco.mjv_makeConnector(
        scene.geoms[scene.ngeom-1], # geom
        mujoco.mjtGeom.mjGEOM_ARROW, # type
        radius, # width
        *point1[:3], # a0 - a2
        *point2[:3], # b0 - b2
    )

def add_sphere_to_mjv_scene(scene : mujoco.MjvScene, point : np.ndarray, radius : float, rgba : np.ndarray):
    if scene.ngeom >= scene.maxgeom:
        return
    
    scene.ngeom += 1  # increment ngeom

    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom-1], # geom
        mujoco.mjtGeom.mjGEOM_SPHERE, # type
        np.array([radius]*3), # size
        point[:3], # pos
        np.zeros(9), # mat
        rgba.astype(np.float32), # mat
    )