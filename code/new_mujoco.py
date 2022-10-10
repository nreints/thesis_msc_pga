import mujoco_py
import numpy as np
import itertools
from create_strings import create_string
import pickle
import torch
from pyquaternion import Quaternion
import roma
from convert import *
import copy

# q = torch.randn(4) # Random unnormalized quaternion
# qconv = roma.quat_conjugation(q) # Quaternion conjugation
# print(q,
# qconv)
# qinv = roma.quat_inverse(q) # Quaternion inverse
# print(roma.quat_product(q, qinv)) # -> [0,0,0,1] identity quaternion

VERT_NUM = 1

# def own_rotVecQuat(v, q):
# original from https://math.stackexchange.com/questions/40164/how-do-you-rotate-a-vector-by-a-unit-quaternion
#     v_new = np.zeros(4)
#     v_new[1:] = v
#     part1 = rot_quaternions(v_new, q)
#     print(part1)
#     q_prime = q
#     q_prime[1:] = -q_prime[1:]
#     print(q_prime, "prime")
#     return rot_quaternions(q_prime, part1)

def fast_rotVecQuat(v, q):
    """
    Returns the rotated batch of vectors v by a batch quaternion q.
    v shape: batchx8x3
    q shape: batchx4
    """
    device = v.device

    q_norm = torch.div(q.T, torch.norm(q, dim=-1)).T

    # print(Quaternion(np.array(q_norm[0].detach())).norm)

    # Swap columns for roma calculations (bi, cj, dk, a)
    q_new1 = torch.index_select(q_norm, 1, torch.tensor([1, 2, 3, 0]).to(device))

    # return roma.quat_action(q_new1, v, is_normalized=False)

    v_test = v.mT

    rot_mat = (roma.unitquat_to_rotmat(q_new1) @ v_test).mT.to(device)

    return rot_mat

# def rot_quaternions(q1, q2):
#     # https://stackoverflow.com/questions/39000758/how-to-multiply-two-quaternions-by-python-or-numpy
#     w0, x0, y0, z0 = q1
#     w1, x1, y1, z1 = q2
#     return torch.tensor([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
#                      x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
#                      -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
#                      x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=torch.float64)

# def own_rotVecQuat(v, q):
#     # According to mujoco? Ask Steven/Leo
#     v_new = torch.zeros(4)
#     v_new[1:] = v
#     part1 = rot_quaternions(q, v_new)
#     q_prime = q
#     q_prime[1:] = -q_prime[1:]
#     return rot_quaternions(part1, q_prime)[1:]


def rotVecQuat(v, q):
    # From internet MuJoCo
    res = np.zeros(3)
    mujoco_py.functions.mju_rotVecQuat(res, v, q)
    return res


# def get_vert_coords_quat(sim, obj_id, xyz_local):
#     """
#     Returns the locations of the vertices during simulation
#         using quaternion
#     """
#     obj_xquat = sim.data.body_xquat[obj_id]
#     trans = sim.data.body_xpos[obj_id]
#     # FIX Geeft nu alleen de geroteerde eerste vertice terug [:,??]
#     return rotVecQuat(xyz_local[:,VERT_NUM], obj_xquat) + trans

def get_quat(sim, obj_id):
    return sim.data.body_xquat[obj_id]

def get_mat(sim, obj_id):
    # TODO Check how the matrix is constructed
    return sim.data.body_xmat[obj_id]

def get_vert_local(sim, obj_id):
    """
    Returns the initial locations of the vertices
    """
    obj_size = sim.model.geom_size[obj_id]
    offsets = np.array([-1, 1]) * obj_size[:, None]
    return np.stack(list(itertools.product(*offsets))).T

def get_vert_coords(sim, obj_id, xyz_local):
    """
    Returns the locations of the vertices during simulation
    """
    # Translation Vector
    obj_pos = sim.data.geom_xpos[obj_id]
    # Rotation Matrix
    obj_mat = sim.data.geom_xmat[obj_id].reshape(3, 3)
    return obj_pos[:, None] + obj_mat @ xyz_local

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


    arccos = np.arccos(quat[0]/norm)
    part2 = inv_norm * arccos * quat[1:]

    logQuat = np.append(log_norm, part2)

    return logQuat

def create_empty_dataset(local_start):
    """
    Returns empyt data dictionary.
    """
    return {
            "start": local_start.T,
            "pos": np.empty((n_steps//10, 8, 3)),
            "eucl_motion" : np.empty((n_steps//10, 1, 12)),
            "quat": np.empty((n_steps//10, 1, 7)),
            "log_quat": np.empty((n_steps//10, 1, 7)),
            "pos_diff": np.empty((n_steps//10, 8, 3)),
            "pos_diff_start": np.empty((n_steps//10, 8, 3)),
            "pos_norm": np.empty((n_steps//10, 8, 3))
            }

def generate_data(string, n_steps, visualize):
    """
    Create the dataset of data_type for n//10 steps.
    """

    geom_name = 'object_geom'

    model = mujoco_py.load_model_from_xml(string)
    sim = mujoco_py.MjSim(model)

    geom_id = model.geom_names.index(geom_name) + 1
    xyz_local = get_vert_local(sim, geom_id)

    dataset = create_empty_dataset(xyz_local)

    if visualize:
        viewer = mujoco_py.MjViewer(sim)

    for i in range(n_steps):
        sim.step()

        if visualize:
            viewer.render()

        if i == 0:
            prev = get_vert_coords(sim, geom_id, xyz_local).T
            start = prev
            start_x_pos = copy.deepcopy(sim.data.body_xpos[geom_id])

            # First difference should be zero
            dataset["pos_diff_start"][i] = np.zeros((8, 3))

        if i % 10 == 0:
            # Collect position data
            dataset["pos"][i//10] = get_vert_coords(sim, geom_id, xyz_local).T
            # print(get_vert_coords(sim, geom_id, xyz_local).T[0,2] - prev[0, 2])
            # if get_vert_coords(sim, geom_id, xyz_local).T[0, 2] > prev[0, 2]:
            #     print("flying away")
            #     exit()

            # Collect euclidean motion data
            dataset["eucl_motion"][i//10] = np.append(get_mat(sim, geom_id), sim.data.body_xpos[geom_id])

            # Collect quaternion data
            dataset["quat"][i//10] = np.append(get_quat(sim, geom_id), sim.data.body_xpos[geom_id])

            # Collect Log Quaternion data
            dataset["log_quat"][i//10] = np.append(calculate_log_quat(get_quat(sim, geom_id)), sim.data.body_xpos[geom_id])

            if i != 0:
                dataset["pos_diff"][i//10] = get_vert_coords(sim, geom_id, xyz_local).T - prev

                prev = get_vert_coords(sim, geom_id, xyz_local).T

                dataset["pos_diff_start"][i//10] = get_vert_coords(sim, geom_id, xyz_local).T - start


    print(len(dataset["quat"]) == len(dataset["pos"]))
    # print(dataset["pos"][10])
    # print(dataset["quat"].squeeze().shape)
    # quat = Quaternion(dataset["quat"][0][0][:4])
    # print(quat.rotate(xyz_local.T[0]) + dataset["quat"][10][0][4:])

    dataset["pos_norm"] = (dataset["pos"] - np.mean(dataset["pos"], axis=(0,1))) / np.std(dataset["pos"], axis=(0,1))
    # print(np.mean(dataset["pos"], axis=(0,1)), np.std(dataset["pos"], axis=(0,1)))
    return dataset


def write_data_nsim(num_sims, n_steps, obj_type, visualize=False):
    for sim_id in range(num_sims):
        print("sim: ", sim_id)
        euler = f"{np.random.uniform(-40, 40)} {np.random.uniform(-40, 40)} {np.random.uniform(-40, 40)}"
        # euler = f"{0.0} {0.0} {0.0}"
        pos = f"{np.random.uniform(-10, 10)} {np.random.uniform(-10, 10)} {np.random.uniform(10, 30)}"
        size = f"{np.random.uniform(0.5, 5)} {np.random.uniform(0.5, 5)} {np.random.uniform(0.5, 5)}"
        # print("size object", size)

        string = create_string(euler, pos, obj_type, size)
        dataset = generate_data(string, n_steps, visualize)

        sim_data = {"vars" : [euler, pos, obj_type, size], "data" : dataset}
        with open(f'data/sim_{sim_id}.pickle', 'wb') as f:
            pickle.dump(sim_data, f)
        f.close()

if __name__ == "__main__":
    ## Uncomment to create random data
    n_sims = 2000
    n_steps = 2250
    obj_type = "box"

    write_data_nsim(n_sims, n_steps, obj_type, visualize=False)