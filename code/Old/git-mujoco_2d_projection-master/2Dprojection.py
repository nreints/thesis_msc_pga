import numpy as np
import matplotlib.pyplot as plt
import mujoco_py
from mujoco_py import load_model_from_path, MjSim, MjViewer
import os

import gym
from gym.envs.robotics.rotations import mat2euler, mat2quat, quat2euler, euler2quat


BASIC_MODEL_XML = """
<mujoco>
    <worldbody>
        <light name="light1" diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
        <camera name="camera1" pos="0 0 0" fovy="90" euler="90 0 0"/>
        <geom name="geom1" pos="0.5 0.4 0" type="plane" size="1 1 0.1" rgba=".9 0 0 1"/>
        <body pos="0 0 1" name="body1">
            <joint name="joint1" type="free"/>
            <geom name="geom2" pos="0 1 0" type="box" size=".1 .2 .3" rgba="0 .9 0 1"/>
            <site name="site1" pos="1 0 0" size="0.1" type="sphere"/>
            <site name="sensorsurf" pos="0 0.045 0" size=".03 .03 .03" type="ellipsoid" rgba="0.3 0.2 0.1 0.3"/>
        </body>
        <body pos="1 0 0" name="mocap1" mocap="true">
            <geom conaffinity="0" contype="0" pos="0 0 0" size="0.01 0.01 0.01" type="box"/>
        </body>
    </worldbody>
    <sensor>
        <touch name="touchsensor" site="sensorsurf" />
    </sensor>
</mujoco>
"""

def global2label(obj_pos, cam_pos, cam_ori, output_size=[64, 64], fov=90, s=1):
    """
    :param obj_pos: 3D coordinates of the joint from MuJoCo in nparray [m]
    :param cam_pos: 3D coordinates of the camera from MuJoCo in nparray [m]
    :param cam_ori: camera 3D rotation (Rotation order of x->y->z) from MuJoCo in nparray [rad]
    :param fov: field of view in integer [degree]
    :return: Heatmap of the object in the 2D pixel space.
    """

    e = np.array([output_size[0]/2, output_size[1]/2, 1])
    fov = np.array([fov])

    # Converting the MuJoCo coordinate into typical computer vision coordinate.
    cam_ori_cv = np.array([cam_ori[1], cam_ori[0], -cam_ori[2]])
    obj_pos_cv = np.array([obj_pos[1], obj_pos[0], -obj_pos[2]])
    cam_pos_cv = np.array([cam_pos[1], cam_pos[0], -cam_pos[2]])

    obj_pos_in_2D, obj_pos_from_cam = get_2D_from_3D(obj_pos_cv, cam_pos_cv, cam_ori_cv, fov, e)
    label = gkern(output_size[0], output_size[1], (obj_pos_in_2D[1], output_size[0]-obj_pos_in_2D[0]), sigma=s)
    return label

def get_2D_from_3D(a, c, theta, fov, e):
    """
    :param a: 3D coordinates of the joint in nparray [m]
    :param c: 3D coordinates of the camera in nparray [m]
    :param theta: camera 3D rotation (Rotation order of x->y->z) in nparray [rad]
    :param fov: field of view in integer [degree]
    :param e: 
    :return:
        - (bx, by) ==> 2D coordinates of the obj [pixel]
        - d ==> 3D coordinates of the joint (relative to the camera) [m]
    """

    # Get the vector from camera to object in global coordinate.
    ac_diff = a - c

    # Rotate the vector in to camera coordinate
    x_rot = np.array([[1 ,0, 0],
                    [0, np.cos(theta[0]), np.sin(theta[0])],
                    [0, -np.sin(theta[0]), np.cos(theta[0])]])

    y_rot = np.array([[np.cos(theta[1]) ,0, -np.sin(theta[1])],
                [0, 1, 0],
                [np.sin(theta[1]), 0, np.cos(theta[1])]])

    z_rot = np.array([[np.cos(theta[2]) ,np.sin(theta[2]), 0],
                [-np.sin(theta[2]), np.cos(theta[2]), 0],
                [0, 0, 1]])

    transform = z_rot.dot(y_rot.dot(x_rot))
    d = transform.dot(ac_diff)

    # scaling of projection plane using fov
    fov_rad = np.deg2rad(fov)
    e[2] *= e[1]*1/np.tan(fov_rad/2.0)

    # Projection from d to 2D
    print(e[2], d[0], (d[2]), e[0])
    bx = e[2]*d[0]/(d[2]) + e[0]
    by = e[2]*d[1]/(d[2]) + e[1]

    return (bx, by), d

def gkern(h, w, center, sigma=1):
    x = np.arange(0, w, 1, float)
    y = np.arange(0, h, 1, float)
    y = y[:, np.newaxis]
    x0 = center[0]
    y0 = center[1]
    return np.exp(-1 * ((x - x0) ** 2 + (y - y0) ** 2) / sigma**2)

if __name__ == "__main__":

    model = mujoco_py.load_model_from_xml(BASIC_MODEL_XML)
    sim = MjSim(model)
    with open('hans.xml', 'w') as f:
        sim.save(f)
    data = MjData(model)
    sim.forward()
    viewer = MjViewer(sim)

    print(sim.model.body_names)
    print(sim.model.geom_names)

    # Use self.sim.data.get_geom_xpos("object") to get the object pos (mujoco-py env)
    obj_pos = sim.data.get_geom_xpos("geom2")
    # obj_pos = np.array([0.2, 0.25, 1.0])

    # Use env.sim.data.get_camera_xpos('camera') to get the camera pos (mujoco-py env)
    cam_pos = sim.data.get_camera_xpos("camera1")
    # cam_pos = np.array([0.7, 0, 1.5])

    # Use gym.envs.robotics.rotations.mat2euler(env.sim.data.get_camera_xmat('camera1')) to get the camera orientation in euler form. (mujoco-py + gym)       
    cam_ori = gym.envs.robotics.rotations.mat2euler(sim.data.get_camera_xmat("camera1"))
    # cam_ori = np.array([0.2, 1.2, 1.57])

    fov = 90 # Field of view of the camera
    output_size = [128, 128] # Output size (Height and width) of the 2D projection label in pixel
    s = 0.5 # std for heapmap signal

    print(obj_pos)
    print(cam_pos)
    print(cam_ori)

    label = global2label(obj_pos, cam_pos, cam_ori, output_size, fov=fov, s=s)
    print("label", label)
    plt.imshow(label)
    plt.show()
