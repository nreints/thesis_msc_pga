import mujoco
import numpy as np
import itertools
from create_strings import create_string
import pickle
from convert import *
from pyquaternion import Quaternion
# import mujoco_viewer
import random
import os
import argparse
import time

def get_mat(data, obj_id):
    """
    Returns the rotation matrix of an object.
    """
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
    # a bi cj dk convention (identity: 1 0 0 0)
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

def generate_data(string, n_steps, visualize=False, qvel_range_t=(0,0), qvel_range_r=(0,0)):
    """
    Create the dataset of data_type for n//10 steps.
    """

    geom_name = "object_geom"

    model = mujoco.MjModel.from_xml_string(string)

    data = mujoco.MjData(model)
    # qvel 012 -> linear velocity
    # qvel 345 -> angular velocity
    # Set random initial velocities
    # TODO
    data.qvel[0:3] = np.random.rand(3) * [random.randint(qvel_range_t[0], qvel_range_t[1]) for _ in range(3)]
    data.qvel[3:6] = np.random.rand(3) * [random.randint(qvel_range_r[0], qvel_range_r[1]) for _ in range(3)]
    geom_id = model.geom(geom_name).id

    xyz_local = get_vert_local(model, geom_id)

    dataset = create_empty_dataset(xyz_local)

    if visualize:
        import mujoco_viewer
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

                # Collect euclidean motion data
                dataset["eucl_motion"][i // 10] = np.append(
                    get_mat(data, geom_id), xpos
                )

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

    # if visualize:
    #     viewer.close()

    return dataset


def write_data_nsim(num_sims, n_steps, obj_type, symmetry, visualize=False, qvel_range_t=(0,0), qvel_range_r=(0,0)):
    
    dir = f"data/data_t{qvel_range_t}_r{qvel_range_r}_{symmetry}"
    if not os.path.exists("data"):
        os.mkdir("data")
    if not os.path.exists(dir):
            print("Creating directory")
            os.mkdir(dir) #TODO mkdir args
    elif len(os.listdir(dir)) > num_sims:
        print(f"This directory already existed with {len(os.listdir(dir))} files, you want {num_sims} files. Please delete directory.")
        raise IndexError(f"This directory ({dir}) already exists with less simulations.")

    for sim_id in range(num_sims):
        if sim_id % 100 == 0 or sim_id == num_sims-1:
            print(f"sim: {sim_id}/{num_sims-1}")
        euler = f"{np.random.uniform(-40, 40)} {np.random.uniform(-40, 40)} {np.random.uniform(-40, 40)}"

        if symmetry == "full":
            size = np.random.uniform(0.5, 5)
            sizes = f"{size} {size} {size}"
        elif symmetry == "semi": #TODO think whether it needs to be more random
            size01 = np.random.uniform(0.5, 5)
            size2 = np.random.uniform(size01+1e-1, 5+size01)
            sizes = f"{size01} {size01} {size2}" #TODO random volgorde list shuffle
        elif symmetry == "none":
            sizes = f"{np.random.uniform(0.5, 5)} {np.random.uniform(0.5, 5)} {np.random.uniform(0.5, 5)}"
        else:
            raise argparse.ArgumentError(f"Not a valid string for argument symmetry: {symmetry}") #TODO baseExeption

        #TODO fix no flying boxes
        pos = f"{np.random.uniform(-10, 10)} {np.random.uniform(-10, 10)} {np.random.uniform(10, 30)}"
        string = create_string(euler, pos, obj_type, sizes)
        dataset = generate_data(string, n_steps, visualize, qvel_range_t, qvel_range_r)

        sim_data = {"vars": [euler, pos, obj_type, sizes], "data": dataset}
        # Create directory if not yet present
        with open(f"{dir}/sim_{sim_id}.pickle", "wb") as f:
            pickle.dump(sim_data, f)
        f.close()

if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("-n_sims", type=int, help="number of simulations", default=1000)
    parser.add_argument("-n_frames", type=int, help="number of frames", default=2000)
    parser.add_argument("-symmetry", type=str, help="symmetry of the box.\nfull: symmetric box\n; semi: 2 sides of same length, other longer\n;none: random lengths", default="none")
    parser.add_argument("-t_min", type=int, help="translation qvel min", default=0)
    parser.add_argument("-t_max", type=int, help="translation qvel max", default=0)
    parser.add_argument("-r_min", type=int, help="rotation qvel min", default=0)
    parser.add_argument("-r_max", type=int, help="rotation qvel max", default=0)
    args = parser.parse_args()
    ## Create random data
    n_sims = args.n_sims
    n_steps = args.n_frames
    t_min = args.t_min
    t_max = args.t_max
    r_min = args.r_min
    r_max = args.r_max
    print(f"Creating dataset qvel_range_t=({t_min}, {t_max}), qvel_range_r=({r_min}, {r_max})")
    obj_type = "box"
    # print(f"qvel_range_t=({t_min}, {t_max}), qvel_range_r=({r_min}, {r_max})")
    write_data_nsim(n_sims, n_steps, obj_type, args.symmetry, visualize=False, qvel_range_t=(t_min,t_max), qvel_range_r=(r_min,r_max))

    print(f"\nTime: {time.time()- start_time}\n---- FINISHED ----")

    # write_data_nsim(n_sims, n_steps, obj_type, visualize=False, qvel_range_t=(t_min,t_max), qvel_range_r=(0,0))
