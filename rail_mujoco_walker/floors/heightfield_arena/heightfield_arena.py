from pathlib import Path
import numpy as np
import scipy
from PIL import Image
import os
import struct

from dm_control import composer
from dm_control.mujoco.wrapper.mjbindings import mjlib
from dm_control import mjcf

_FLOOR_ASSETS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "resources", "floor_assets")
_ARENA_XML_PATH = os.path.join(_FLOOR_ASSETS_PATH, 'heightfield_arena.xml')
_ARENA_HEIGHTMAP_PATH = os.path.join(_FLOOR_ASSETS_PATH, 'textures', 'heightmap.bin')
_ARENA_TEXTURE_PATH = os.path.join(_FLOOR_ASSETS_PATH, 'textures', 'texture_baked_large.png')

class HeightFieldArena(composer.Arena):
    def __init__(self, scale, height_scale=5, name='heightfield_arena'):
        self.scale = scale
        self.height_scale = height_scale
        self.texture_path = _ARENA_TEXTURE_PATH
        self.heightmap_path = _ARENA_HEIGHTMAP_PATH

        # We can read in the heightmap and store it as a numpy array in self.img and self.img_size
        # with open(self.heightmap_path, 'rb') as f:
        #     self.img_size = struct.unpack('2I', f.read(8))
        #     num_pixels = self.img_size[0]*self.img_size[1]
        #     self.img = np.array(struct.unpack(f'{num_pixels}f', f.read(4*num_pixels))).reshape(self.img_size)
        #     self.img = np.flip(self.img, axis=1)
        #     self.img = (self.img - self.img.min()) / (self.img.max() - self.img.min()) * self.scale / 5
        #     print("Height field size", self.img_size)

        super(HeightFieldArena, self).__init__(name=name)

        self._mjcf_root.visual.scale.framelength = 10
        self._mjcf_root.visual.scale.framewidth = 5

    def _build(self, name):
        # Don't call super()._build here because we want our own XML file
        self._mjcf_root = mjcf.from_path(_ARENA_XML_PATH)
        if name:
            self._mjcf_root.model = name

        self._mjcf_root.option.gravity = (0, 0, -9.81)

        # =========================== Init heightfield and texturize ===========================

        self._hfield = self._mjcf_root.asset.add('hfield',
                                                 name='terrain',
                                                 file=f'{self.heightmap_path}',
                                                 size=(self.scale, self.scale, self.scale / self.height_scale, 0.1))

        self._texture = self._mjcf_root.asset.add('texture',
                                                  name='aesthetic_texture',
                                                  file=f'{self.texture_path}',
                                                  type='2d')

        self._material = self._mjcf_root.asset.add('material',
                                                   texrepeat=(1, 1),
                                                   name='aesthetic_material',
                                                   texture=self._texture,
                                                   texuniform='false')

        self._terrain_geom = self._mjcf_root.worldbody.add(
            'geom',
            name='terrain',
            type='hfield',
            rgba=(1.0, 1.0, 1.0, 1.0),
            pos=(0, 0, -0.01),
            hfield='terrain',
            material=self._material)


        # =========================== Add camera ===========================
        # self._mjcf_root.worldbody.add('camera',
        #                               mode="fixed",
        #                               pos="0 0 25",
        #                               euler="0 0 0")


        # =========================== Add walls to limit robot movement ===========================
        self.walls = []

        self.walls.append(
            self._mjcf_root.worldbody.add('geom',
                                          type="box",
                                          contype="1",
                                          conaffinity="1",
                                          size=(0.1, self.scale + 1, 0.7),
                                          rgba=(0.3, 0.3, 0.9, 0.5),
                                          pos=(self.scale, 0, 0)))

        self.walls.append(
            self._mjcf_root.worldbody.add('geom',
                                          type="box",
                                          contype="1",
                                          conaffinity="1",
                                          size=(0.1, self.scale + 1, 0.7),
                                          rgba=(0.3, 0.3, 0.9, 0.5),
                                          pos=(-self.scale, 0, 0)))

        self.walls.append(
            self._mjcf_root.worldbody.add('geom',
                                          type="box",
                                          contype="1",
                                          conaffinity="1",
                                          size=(self.scale + 1, 0.1, 0.7),
                                          rgba=(0.3, 0.3, 0.9, 0.5),
                                          pos=(0, self.scale, 0)))

        self.walls.append(
            self._mjcf_root.worldbody.add('geom',
                                          type="box",
                                          contype="1",
                                          conaffinity="1",
                                          size=(self.scale + 1, 0.1, 0.7),
                                          rgba=(0.3, 0.3, 0.9, 0.5),
                                          pos=(0, -self.scale, 0)))

    def height_lookup(self, pos):
        """Returns the height of the terrain at the given position."""
        return 0.15

    def in_bounds(self, pos):
        eps = 1.0
        return np.abs(pos).max() < self.scale + eps

