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
import copy
# from tqdm import trange

def get_mat(data, obj_id):
    """
    Returns the rotation matrix of an object.
    """
    return data.geom_xmat[obj_id].reshape(3,3)

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
    # TODO WATCH OUT!!! If plane -> [obj_id]
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
        "eucl_motion_old": np.empty((n_steps // 10, 1, 12)),
        "quat": np.empty((n_steps // 10, 1, 7)),
        "quat_old": np.empty((n_steps // 10, 1, 7)),
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
    Create the dataset of data_type for n_steps//10 steps.
    """
    geom_name = "object_geom"

    model = mujoco.MjModel.from_xml_string(string)

    data = mujoco.MjData(model)
    # qvel 012 -> linear velocity
    # qvel 345 -> angular velocity
    # Set random initial velocities
    data.qvel[0:3] = np.random.uniform(qvel_range_t[0], qvel_range_t[1]+1e-10, size=3)
    # data.qvel[0:3] = [0, 3, 0]
    data.qvel[3:6] = np.random.uniform(qvel_range_r[0], qvel_range_r[1]+1e-10, size=3)
    # data.qvel[3:6] = [0, 40, 0]


    geom_id = model.geom(geom_name).id
    body_id = model.body("object_body").id

    xyz_local = get_vert_local(model, geom_id)

    dataset = create_empty_dataset(xyz_local)

    if visualize:
        import mujoco_viewer
        viewer = mujoco_viewer.MujocoViewer(model, data)

    for i in range(0, n_steps, 10):
        if not visualize or viewer.is_alive:
            mujoco.mj_step(model, data)

            if visualize and (i%5==0 or i==0):

                viewer.render()

            if i == 0:

                start_xpos = copy.deepcopy(data.geom_xpos[geom_id])


                prev_xyz = get_vert_coords(data, geom_id, xyz_local).T
                start_xyz = prev_xyz

                # First difference should be zero
                dataset["pos_diff_start"][i] = np.zeros((8, 3))

                start_rotMat = copy.deepcopy(get_mat(data, geom_id))
                dataset["eucl_motion"][i] = np.append(np.eye(3), np.zeros(3))

                start_quat = copy.deepcopy(get_quat(data, body_id))

            xpos = data.geom_xpos[geom_id]

            # Collect position data
            dataset["pos"][i // 10] = get_vert_coords(data, geom_id, xyz_local).T

            if i >= 10:
                # Collect euclidean motion data
                current_rotMat = get_mat(data, geom_id)

                rel_trans = xpos - current_rotMat @ np.linalg.inv(start_rotMat) @ start_xpos
                rel_rot = current_rotMat @ np.linalg.inv(start_rotMat)

                dataset["eucl_motion"][i // 10] = np.append(
                    rel_rot.flatten(), rel_trans
                )

                quaternion = Quaternion(matrix=rel_rot)
                quat2 = quaternion.elements
                quat2 = (Quaternion(get_quat(data, body_id)) * Quaternion(start_quat).inverse).elements

                dataset["quat"][i // 10] = np.append(
                    quat2, rel_trans
                )

                # TODO log_quat dual_quat log_dual_quat -> FIX START POS OOK IN CONVERT AANROEP

            dataset["eucl_motion_old"][i // 10] = np.append(
                    get_mat(data, geom_id).flatten(), xpos
                )

            # # Quaternion w ai bj ck convention
            quatzzz = get_quat(data, body_id)

            # Collect quaternion data
            dataset["quat_old"][i // 10] = np.append(
                quatzzz, xpos
            )

            # Collect Log Quaternion data
            dataset["log_quat"][i // 10] = np.append(
                calculate_log_quat(quatzzz), xpos
            )

            dualQuaternion = get_dualQ(
                quatzzz, xpos
            )

            # Collect Dual-Quaternion data
            dataset["dual_quat"][i // 10] = dualQuaternion

            # Collect exp_dualQ data
            dataset["log_dualQ"][i // 10] = logDual(dualQuaternion)

            if i != 0:
                dataset["pos_diff"][i // 10] = (
                    get_vert_coords(data, geom_id, xyz_local).T - prev_xyz
                )

                prev_xyz = get_vert_coords(data, geom_id, xyz_local).T

                dataset["pos_diff_start"][i // 10] = (
                    get_vert_coords(data, geom_id, xyz_local).T - start_xyz
                )
        else:
            break

    dataset["pos_norm"] = (
        dataset["pos"] - np.mean(dataset["pos"], axis=(0, 1))
    ) / np.std(dataset["pos"], axis=(0, 1))

    if visualize:
        viewer.close()

    return dataset

def get_sizes(symmetry):
    if symmetry == "full":
        size = np.random.uniform(5, 10)
        return f"{size} {size} {size}"
    elif symmetry == "semi": #TODO think whether it needs to be more random
        ratio = np.array([1,1,10])
        size01 = np.random.uniform(0.5, 5)
        sizes = ratio * size01
        return f"{sizes[0]} {sizes[1]} {sizes[2]}" #TODO random volgorde list shuffle
    elif symmetry == "tennis0":
        # TODO FLYING Quadrilaterally-faced hexahedrons Not necessary if no plane
        ratio = np.array([1, 3, 10])
        random_size = np.random.uniform(0.2, 2)
        sizes = ratio * random_size
        return f"{sizes[0]} {sizes[1]} {sizes[2]}"
    elif symmetry == "tennis1":
        ratio = np.array([1, 2, 3])
        random_size = np.random.uniform(0.5, 1.5)
        sizes = ratio * random_size
        return f"{sizes[0]} {sizes[1]} {sizes[2]}"
    elif symmetry == "none":
        return f"{np.random.uniform(0.5, 5)} {np.random.uniform(0.5, 5)} {np.random.uniform(0.5, 5)}"
    else:
        raise argparse.ArgumentError(f"Not a valid string for argument symmetry: {symmetry}") #TODO baseExeption

def get_dir(qvel_range_t, qvel_range_r, symmetry, num_sims, plane, grav):
    dir = f"data/data_t{qvel_range_t}_r{qvel_range_r}_{symmetry}_p{plane}_g{grav}"
    if not os.path.exists("data"):
        os.mkdir("data")
    if not os.path.exists(dir):
            print("Creating directory")
            os.mkdir(dir)
    elif len(os.listdir(dir)) > num_sims:
        print(f"This directory already existed with {len(os.listdir(dir))} files, you want {num_sims} files. Please delete directory.")
        raise IndexError(f"This directory ({dir}) already exists with less simulations.")
    return dir

def write_data_nsim(num_sims, n_steps, symmetry, gravity, dir, visualize=False, qvel_range_t=(0,0), qvel_range_r=(0,0), plane=False):

    for sim_id in range(num_sims):
        if sim_id % 100 == 0 or sim_id == num_sims:
            print(f"Generating sim {sim_id}/{num_sims}")
        # Define euler angle
        euler = f"{np.random.uniform(0, 360)} {np.random.uniform(0, 360)} {np.random.uniform(0, 360)}"
        euler = "0 0 0"
        # Define sizes
        sizes = get_sizes(symmetry)
        # Define position TODO fix no flying Quadrilaterally-faced hexahedrons
        pos = f"{np.random.uniform(-10, 10)} {np.random.uniform(-10, 10)} {np.random.uniform(5, 10)}"
        # pos = "0 0 0"


        string = create_string(euler, pos, sizes, gravity, plane)
        # Create dataset
        dataset = generate_data(string, n_steps, visualize, qvel_range_t, qvel_range_r)
        sim_data = {"vars": {"euler":euler, "pos":pos, "sizes":sizes, "gravity":gravity, "n_steps":n_steps//10}, "data": dataset}
        with open(f"{dir}/sim_{sim_id}.pickle", "wb") as f:
            pickle.dump(sim_data, f)
        f.close()

if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("-n_sims", type=int, help="number of simulations", default=100)
    parser.add_argument("-n_frames", type=int, help="number of frames", default=6000)
    parser.add_argument("-symmetry", type=str, help="symmetry of the box.\nfull: symmetric box\n; semi: 2 sides of same length, other longer\n;tennis0: tennis_racket effect 1,3,10\n;tennis1: tennis_racket effect 1,2,3\n;none: random lengths for each side", default="full")
    parser.add_argument("-t_min", type=int, help="translation qvel min", default=0)
    parser.add_argument("-t_max", type=int, help="translation qvel max", default=0)
    parser.add_argument("-r_min", type=int, help="rotation qvel min", default=4)
    parser.add_argument("-r_max", type=int, help="rotation qvel max", default=5)
    parser.add_argument('--gravity', action=argparse.BooleanOptionalAction)
    parser.add_argument('--plane', action=argparse.BooleanOptionalAction)
    parser.add_argument('--visualize', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    ## Create random data
    n_sims = args.n_sims
    n_steps = args.n_frames

    qvel_range_t=(args.t_min,args.t_max)
    qvel_range_r=(args.r_min,args.r_max)
    print(f"Creating dataset qvel_range_t={qvel_range_t}, qvel_range_r={qvel_range_r}, symmetry={args.symmetry}")


    dir = get_dir(qvel_range_t, qvel_range_r, args.symmetry, args.n_sims, args.plane, args.gravity)

    # print(f"qvel_range_t=({t_min}, {t_max}), qvel_range_r=({r_min}, {r_max})")
    write_data_nsim(n_sims, n_steps, args.symmetry, args.gravity, dir, args.visualize, qvel_range_t, qvel_range_r, args.plane)

    print(f"\nTime: {time.time()- start_time}\n---- FINISHED ----")

    # write_data_nsim(n_sims, n_steps, obj_type, visualize=False, qvel_range_t=(t_min,t_max), qvel_range_r=(0,0))
