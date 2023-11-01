import sys
import os
import platform

current_dir = os.path.dirname(os.path.abspath(__file__))
#project_root_dir = os.path.dirname(current_dir)

machine_arch = platform.machine().lower()
if machine_arch == "x86_64":
    machine_arch = "amd64"

to_append = os.path.join(current_dir, "lib", "python", machine_arch)
#print("appending {} to sys.path".format(to_append))

sys.path.append(
    to_append
)

import robot_interface