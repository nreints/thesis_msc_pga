import mujoco
import numpy as np
import itertools
from create_strings import create_string
import pickle
from convert import *
from pyquaternion import Quaternion
import mujoco_viewer
import random
import os



def get_mat(data, obj_id):
    """
    Returns the rotation matrix of an object.
    """
    # Correct reshape according to converter.
    return data.xmat[obj_id]

def get_vert_local(model, obj_id):
    """
    Returns the locations of the vertices centered around zero.
    """
    obj_size = model.geom_size[obj_id]
    offsets = np.array([-1, 1]) * obj_size[:, None]
    return np.stack(list(itertools.product(*offsets))).T


def get_vert_coords(data, obj_id, xyz_local):
    """
    Returns the locations of the vertices during simulation
    """
    # Translation vector
    obj_pos = data.geom_xpos[obj_id]

    # Rotation Matrix
    obj_mat = data.geom_xmat[obj_id].reshape(3, 3)

    # R @ v + t
    return obj_mat @ xyz_local + obj_pos[:, None]

def get_quat(data, obj_id):
    """
    Returns the quaternion of an object.
    """
    # MUJOCO DOCS Cartesian orientation of body frame 
    # a bi cj dk convention (when no rotation 1 0 0 0)
    return data.xquat[obj_id]

def calculate_log_quat(quat):
    """
    Calculate the log quaternion based on the quaternion according
        to https://en.wikipedia.org/wiki/Quaternion#Exponential,_logarithm,_and_power_functions
    """
    norm = np.linalg.norm(quat)
    log_norm = np.log(norm)

    # TODO np.linalg.norm(quat[1:]) = 0
    if np.linalg.norm(quat[1:]) == 0:
        inv_norm = 0
    else:
        inv_norm = 1 / np.linalg.norm(quat[1:])

    arccos = np.arccos(quat[0] / norm)
    part2 = inv_norm * arccos * quat[1:]

    logQuat = np.append(log_norm, part2)

    return logQuat

def get_dualQ(quat, translation):
    """
    Returns the dualquaternion of an object.
    """
    # https://cs.gmu.edu/~jmlien/teaching/cs451/uploads/Main/dual-quaternion.pdf
    qr = quat

    t = np.hstack((np.array([0]), translation))

    # if qr[0] == 1 & (qr[1] == qr[2] == qr[3] == 0):
    #     qd = t
    # else:
    qd = (0.5 * Quaternion(t) * Quaternion(quat)).elements

    dual = np.append(qr, qd)
    return dual

def logDual(r):
    """
    Input rotor (8 numbers) returns bivector (=log of rotor) (6 numbers)
    (14 mul, 5 add, 1 div, 1 acos, 1 sqrt)
    """
    if r[0] == 1:
        return np.array([-r[5], -r[6], -r[7], 0, 0, 0])
    a = 1 / (1 - r[0] * r[0])
    b = np.arccos(r[0]) * np.sqrt(a)
    c = a * r[4] * (1 - r[0] * b)
    return np.array([
                    c * r[1] - b * r[5],
                    c * r[2] - b * r[6],
                    c * r[3] - b * r[7],
                    b * r[3],
                    b * r[2],
                    b * r[1]
                ])

def create_empty_dataset(local_start):
    """
    Returns empyt data dictionary.
    """
    return {
        "start": local_start.T,
        "pos": np.empty((n_steps // 10, 8, 3)),
        "eucl_motion": np.empty((n_steps // 10, 1, 12)),
        "quat": np.empty((n_steps // 10, 1, 7)),
        "log_quat": np.empty((n_steps // 10, 1, 7)),
        "dual_quat": np.empty((n_steps // 10, 1, 8)),
        "pos_diff": np.empty((n_steps // 10, 8, 3)),
        "pos_diff_start": np.empty((n_steps // 10, 8, 3)),
        "pos_norm": np.empty((n_steps // 10, 8, 3)),
        "trans": np.empty((n_steps // 10, 3)),
        "log_dualQ": np.empty((n_steps // 10, 6))
    }


def generate_data(string, n_steps, visualize=False):
    """
    Create the dataset of data_type for n//10 steps.
    """

    geom_name = "object_geom"

    model = mujoco.MjModel.from_xml_string(string)

    data = mujoco.MjData(model)
    # qvel 012 -> translational
    # qvel 345 -> rotational
    # Set random initial velocity
    # TODO
    # data.qvel = np.random.rand(6) * random.randint(-5, 5)
    geom_id = model.geom(geom_name).id

    xyz_local = get_vert_local(model, geom_id)
    # print("local", xyz_local.T)

    dataset = create_empty_dataset(xyz_local)

    if visualize:
        viewer = mujoco_viewer.MujocoViewer(model, data)

    for i in range(n_steps):

        if not visualize or viewer.is_alive:
            mujoco.mj_step(model, data)

            if visualize and (i%5==0 or i==0):
                viewer.render()

            if i == 0:

                prev = get_vert_coords(data, geom_id, xyz_local).T
                start = prev

                # First difference should be zero
                dataset["pos_diff_start"][i] = np.zeros((8, 3))

            if i % 10 == 0:

                xpos = data.geom_xpos[geom_id]

                # Collect position data
                dataset["pos"][i // 10] = get_vert_coords(data, geom_id, xyz_local).T

                position = get_vert_coords(data, geom_id, xyz_local).T
                # print("local", xyz_local.T)
                # print(abs(max(xyz_local.T[:,0])- min(xyz_local.T[:,0])))
                # print(abs(max(xyz_local.T[:,1])- min(xyz_local.T[:,1])))
                # print(abs(max(xyz_local.T[:,2])- min(xyz_local.T[:,2])))
                # print("pos", get_vert_coords(data, geom_id, xyz_local).T)
                # print(abs(max(position[:,0])- min(position[:,0])))
                # print(abs(max(position[:,1])- min(position[:,1])))
                # print(abs(max(position[:,2])- min(position[:,2])))
                # exit()

                # Collect euclidean motion data
                dataset["eucl_motion"][i // 10] = np.append(
                    get_mat(data, geom_id), xpos
                )
                # print("Rotation Mat\n", 
                #     get_mat(data, geom_id).reshape(3,3))

                # Quaternion w ai bj ck convention
                quaternion = get_quat(data, geom_id)

                # Collect quaternion data
                dataset["quat"][i // 10] = np.append(
                    quaternion, xpos
                )

                # Collect Log Quaternion data
                dataset["log_quat"][i // 10] = np.append(
                    calculate_log_quat(quaternion), xpos
                )

                dualQuaternion = get_dualQ(
                    quaternion, xpos
                )

                # Collect Dual-Quaternion data
                dataset["dual_quat"][i // 10] = dualQuaternion

                # Collect exp_dualQ data
                dataset["log_dualQ"][i // 10] = logDual(dualQuaternion)

                if i != 0:
                    dataset["pos_diff"][i // 10] = (
                        get_vert_coords(data, geom_id, xyz_local).T - prev
                    )

                    prev = get_vert_coords(data, geom_id, xyz_local).T

                    dataset["pos_diff_start"][i // 10] = (
                        get_vert_coords(data, geom_id, xyz_local).T - start
                    )
        else:
            break

    dataset["pos_norm"] = (
        dataset["pos"] - np.mean(dataset["pos"], axis=(0, 1))
    ) / np.std(dataset["pos"], axis=(0, 1))

    if visualize:
        viewer.close()

    return dataset


def write_data_nsim(num_sims, n_steps, obj_type, visualize=False):
    for sim_id in range(num_sims):
        if sim_id % 10 == 0 or sim_id == num_sims-1:
            print(f"sim: {sim_id}/{num_sims-1}")
        euler = f"{np.random.uniform(-40, 40)} {np.random.uniform(-40, 40)} {np.random.uniform(-40, 40)}"
        euler = "0 0 0"
        pos = f"{np.random.uniform(-10, 10)} {np.random.uniform(-10, 10)} {np.random.uniform(10, 30)}"
        # pos = "10 10 10"
        size = f"{np.random.uniform(0.5, 5)} {np.random.uniform(0.5, 5)} {np.random.uniform(0.5, 5)}"
        size = "3 6 18"
        # print(euler)
        string = create_string(euler, pos, obj_type, size)
        dataset = generate_data(string, n_steps, visualize)

        sim_data = {"vars": [euler, pos, obj_type, size], "data": dataset}

        # Create directory if not yet present
        if not os.path.exists("data"):
            os.mkdir("data")
        with open(f"data/sim_{sim_id}.pickle", "wb") as f:
            pickle.dump(sim_data, f)
        f.close()



if __name__ == "__main__":
    ## Create random data
    n_sims = 50
    n_steps = 5000
    obj_type = "box"

    write_data_nsim(n_sims, n_steps, obj_type, visualize=False)
