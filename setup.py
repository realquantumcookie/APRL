# setup.py
from setuptools import setup

setup(
    name='rail_unified_walker',
    version='0.0.1',
    packages=['rail_walker_interface', "rail_mujoco_walker", "rail_real_walker", "unitree_go1_wrapper", "rail_walker_gym", "jaxrl5"]
)