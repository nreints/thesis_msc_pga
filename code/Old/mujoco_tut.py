import mujoco_py
import numpy as np
import itertools
import matplotlib.pyplot as plt
import PIL.Image

# The basic mujoco wrapper.
from dm_control import mujoco

# Access to enums and MuJoCo library functions.
from dm_control.mujoco.wrapper.mjbindings import enums
from dm_control.mujoco.wrapper.mjbindings import mjlib

# PyMJCF
from dm_control import mjcf

# Composer high level imports
from dm_control import composer
from dm_control.composer.observation import observable
from dm_control.composer import variation

# Imports for Composer tutorial example
from dm_control.composer.variation import distributions
from dm_control.composer.variation import noises
from dm_control.locomotion.arenas import floors

# Control Suite
from dm_control import suite

# Run through corridor example
from dm_control.locomotion.walkers import cmu_humanoid
from dm_control.locomotion.arenas import corridors as corridor_arenas
from dm_control.locomotion.tasks import corridors as corridor_tasks

# Soccer
from dm_control.locomotion import soccer

# Manipulation
from dm_control import manipulation

from dm_control import mujoco

swinging_body = """
<mujoco>
  <worldbody>
    <light name="top" pos="0 0 1"/>
    <body name="box_and_sphere" euler="0 0 -30">
      <joint name="swing" type="hinge" axis="1 -1 0" pos="-.2 -.2 -.2"/>
      <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""

physics = mujoco.Physics.from_xml_string(swinging_body)
# Visualize the joint axis.
scene_option = mujoco.wrapper.core.MjvOption()
scene_option.flags[enums.mjtVisFlag.mjVIS_JOINT] = True
pixels = physics.render(scene_option=scene_option)
PIL.Image.fromarray(pixels)

# Get the world coordinates of the box corners
box_pos = physics.named.data.geom_xpos['red_box']
box_mat = physics.named.data.geom_xmat['red_box'].reshape(3, 3)
box_size = physics.named.model.geom_size['red_box']
offsets = np.array([-1, 1]) * box_size[:, None]
xyz_local = np.stack(itertools.product(*offsets)).T
xyz_global = box_pos[:, None] + box_mat @ xyz_local

# Camera matrices multiply homogenous [x, y, z, 1] vectors.
corners_homogeneous = np.ones((4, xyz_global.shape[1]), dtype=float)
corners_homogeneous[:3, :] = xyz_global

# Get the camera matrix.
camera = mujoco.Camera(physics)
camera_matrix = camera.matrix

# Project world coordinates into pixel space. See:
# https://en.wikipedia.org/wiki/3D_projection#Mathematical_formula
xs, ys, s = camera_matrix @ corners_homogeneous
# x and y are in the pixel coordinate system.
x = xs / s
y = ys / s

# Render the camera view and overlay the projected corner coordinates.
pixels = camera.render()
fig, ax = plt.subplots(1, 1)
ax.imshow(pixels)
ax.plot(x, y, '+', c='w')
ax.set_axis_off()
plt.show()















# model = mujoco_py.load_model_from_xml(swinging_body)

# sim = mujoco_py.MjSim(model)
# sim.reset()
# viewer = mujoco_py.MjViewer(sim)

# # for i in range(1000):
# #    sim.step()
# #    viewer.render()


# # sim.reset()


# # Get the world coordinates of the box corners
# index_red_box = model.geom_names.index('red_box')
# box_pos = sim.data.geom_xpos[index_red_box]
# print(box_pos)
# box_mat = sim.data.geom_xmat[index_red_box].reshape(3, 3)
# print(box_mat)
# box_size = sim.model.geom_size[index_red_box]

# offsets = np.array([-1, 1]) * box_size[:, None]
# xyz_local = np.stack(itertools.product(*offsets)).T
# xyz_global = box_pos[:, None] + box_mat @ xyz_local

# # Camera matrices multiply homogenous [x, y, z, 1] vectors.
# corners_homogeneous = np.ones((4, xyz_global.shape[1]), dtype=float)
# corners_homogeneous[:3, :] = xyz_global

# # Get the camera matrix.
# camera = mujoco.Camera(physics)
# # camera_matrix = camera.matrix
# camera_matrix = sim.data.cam_xmat

# # Project world coordinates into pixel space. See:
# # https://en.wikipedia.org/wiki/3D_projection#Mathematical_formula
# print(camera_matrix.shape, corners_homogeneous.shape)
# xs, ys, s = camera_matrix @ corners_homogeneous
# # x and y are in the pixel coordinate system.
# x = xs / s
# y = ys / s

# # Render the camera view and overlay the projected corner coordinates.
# pixels = viewer.render()
# # pixels = camera.render()
# fig, ax = plt.subplots(1, 1)
# ax.imshow(pixels)
# ax.plot(x, y, '+', c='w')
# ax.set_axis_off()
