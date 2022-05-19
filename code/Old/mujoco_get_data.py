# The basic mujoco wrapper.
# from all_mujoco_test import MODEL_XML_blokkie
from dm_control import mujoco
from dm_control import _render
import xml_strings
import itertools
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
import matplotlib
import mujoco_py
from dm_control import viewer

import matplotlib.animation as animation
from IPython.display import HTML


def load_physics_from_xml(model_name=xml_strings.model_xml_swinging_body):
    return mujoco.Physics.from_xml_string(model_name)


def get_global_coords(physics, object_name):
    box_pos = physics.named.data.geom_xpos[object_name]
    box_mat = physics.named.data.geom_xmat[object_name].reshape(3, 3)
    box_size = physics.named.model.geom_size[object_name]
    offsets = np.array([-1, 1]) * box_size[:, None]
    xyz_local = np.stack(list(itertools.product(*offsets))).T
    # print(xyz_local)
    return box_pos[:, None] + box_mat @ xyz_local


def get_projected_object(physics, object_name=''):
    # Get the world coordinates of the box corners
    xyz_global = get_global_coords(physics, object_name)

    # Camera matrices multiply homogenous [x, y, z, 1] vectors.
    corners_homogeneous = np.ones((4, xyz_global.shape[1]), dtype=float)
    corners_homogeneous[:3, :] = xyz_global

    # Get the camera matrix.
    camera = mujoco.Camera(physics)
    camera_matrix = camera.matrix
    print(camera_matrix)

    # Project world coordinates into pixel space. See:
    # https://en.wikipedia.org/wiki/3D_projection#Mathematical_formula
    xs, ys, s = camera_matrix @ corners_homogeneous
    # x and y are in the pixel coordinate system.
    x = xs / s
    y = ys / s
    return x, y, camera

def plot_corners(x, y, camera):
    pixels = camera.render()
    _, ax = plt.subplots(1, 1)
    ax.imshow(pixels)
    ax.plot(x, y, '+', c='w')
    ax.set_axis_off()
    plt.show()

def show_state(physics):
    pixels = physics.render()
    # print(pixels)
    PIL.Image.fromarray(pixels).show()

def show_corn_time(physics, duration=3, framerate=30):
    frames = []
    physics.reset()  # Reset state and time
    i = 0
    while physics.data.time < duration:
        physics.step()
        i += 1
        if len(frames) < physics.data.time * framerate and i%50==0:
            x, y, camera = get_projected_object(physics, object_name='red_box')
            plot_corners(x, y, camera)
            plt.close()
            # pixels = physics.render()
            # frames.append(pixels)

def save_data(physics, filename, duration=2):
    physics.reset()
    timesteps = int(duration * 500)
    coords_2d = [[] for _ in range(timesteps)]
    i = 0
    while physics.data.time < duration:
        x, y, _ = get_projected_object(physics, object_name='red_box')
        coords_2d[i] += [x, y]
        physics.step()
        i += 1
    with open(filename, 'wb') as f:
        np.save(f, coords_2d)

def load_data(filename):
    with open(filename, 'rb') as f:
        print(np.load(f).shape)



if __name__ == "__main__":
    # physics = load_physics_from_xml(xml_strings.model_xml_blokje)
    # # physics.reset()
    # # show_state(physics)



    # model = mujoco_py.load_model_from_xml(xml_strings.model_xml_blokje)
    # sim = mujoco_py.MjSim(model)
    # viewer = mujoco_py.MjViewer(sim)
    # # print(sim.data.get_camera_xpos(-1))
    # for i in range(10000):
    # #  print(sim.get_state())
    # #  print("vert\t", model.mesh_vert)
    #     sim.step()
    #     viewer.render()
    # sim.reset()


    physics = load_physics_from_xml(xml_strings.model_xml_swinging_body)
    physics.reset()
    x, y, camera = get_projected_object(physics, object_name='red_box')
    plot_corners(x, y, camera)
    # show_state(physics)
    show_corn_time(physics)
    # save_data(physics, 1, "test.npy")



# Volgens Steven heb ne al een manier om de vertices op te halen van de objecten,
# en heb je ook per frame gevonden wat quaternion en translation zijn; en wil je de vertices per frame weten.
# Volgens hem is dat een kwestie van quaternion geconjugeerd toepassen op de vertices
# (daar zou wel een routine voor kunnen bestaan) en dan de translatie toevoegen.