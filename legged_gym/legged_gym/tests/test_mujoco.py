import mujoco
import mujoco.viewer
import os
import time
import numpy as np 

# Path to the Hopper XML (e.g., from Gym's assets)
xml_path = "hopper.xml"  # Must be a valid MuJoCo MJCF file

# Load the model and create data
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

actions = np.zeros(model.nu)

data.ctrl = actions

now = time.time()

mujoco.mj_step(model, data)
print(time.time() - now)


