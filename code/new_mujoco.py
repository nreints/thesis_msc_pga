import mujoco_py
import numpy as np
import itertools
from create_strings import create_string
import pickle
import torch
from pyquaternion import Quaternion
import roma


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
    # print(v.shape, q.shape)
    # Batch of v batchx8x3
    # Batch of q batchx4

    # print(v.reshape((v.shape[0]*v.shape[1], -1)))
    v_reshaped = v.reshape((v.shape[0]*v.shape[1], -1))

    q_new = torch.empty_like(q)
    # q_new = q
    q_new[:, 0:3] = q[:, 1:4]
    q_new[:, 3] = q[:, 0]

    q_new = torch.repeat_interleave(q_new, repeats=8, dim=0)
    # print(torch.zeros(v_reshaped.shape[0],1).shape)
    v_new = torch.hstack((v_reshaped, torch.zeros(v_reshaped.shape[0],1)))
    # print(v_new.shape)
    # print(q_new.shape)
    mult = roma.quat_product(v_new, q_new)
    q_conj = roma.quat_conjugation(q_new)
    mult2 = roma.quat_product(q_conj, mult)

    return mult2[:, :3]

def rot_quaternions(q1, q2):
    # https://stackoverflow.com/questions/39000758/how-to-multiply-two-quaternions-by-python-or-numpy
    w0, x0, y0, z0 = q1
    w1, x1, y1, z1 = q2
    return torch.tensor([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                     x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=torch.float64)

def own_rotVecQuat(v, q):
    # According to mujoco? Ask Steven/Leo
    v_new = torch.zeros(4)
    v_new[1:] = v
    part1 = rot_quaternions(q, v_new)
    q_prime = q
    q_prime[1:] = -q_prime[1:]
    return rot_quaternions(part1, q_prime)[1:]


def rotVecQuat(v, q):
    # From internet
    res = np.zeros(3)
    mujoco_py.functions.mju_rotVecQuat(res, v, q)
    return res

# # testing rotVecQuat vs own_rotVecQuat
# v_big = torch.tensor([[[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]],
#                         [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]]])

# q_big = torch.tensor([[0.3,  0.87, 0.0, 0.707], 
#                         [0.3,  0.87, 0.0, 0.707]])

# # q_big = torch.repeat_interleave(q_big, repeats=8, dim=0)


# v = torch.tensor([1, 0, 0])
# q = torch.tensor([0.3,  0.87, 0.0, 0.707])

# print("fast",fast_rotVecQuat(v_big, q_big))
# print("own",own_rotVecQuat(v, q))
# # print("ori", rotVecQuat(v.astype(np.float64), q.astype(np.float64)))

def get_vert_coords_quat(sim, obj_id, xyz_local):
    """
    Returns the locations of the vertices during simulation
        using quaternion
    """
    obj_xquat = sim.data.body_xquat[obj_id]
    trans = sim.data.body_xpos[obj_id]
    # FIX Geeft nu alleen de geroteerde eerste vertice terug [:,??]
    return rotVecQuat(xyz_local[:,VERT_NUM], obj_xquat) + trans

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
    obj_pos = sim.data.geom_xpos[obj_id]
    obj_mat = sim.data.geom_xmat[obj_id].reshape(3, 3)
    return obj_pos[:, None] + obj_mat @ xyz_local

def calculate_log_quat(quat):
    """
    Calculate the log quaternion based on the quaternion according
        to https://en.wikipedia.org/wiki/Quaternion#Exponential,_logarithm,_and_power_functions
    """

    norm = np.linalg.norm(quat)
    log_norm = np.log(norm)

    inv_norm = 1 / np.linalg.norm(quat[1:])
    arccos = np.arccos(quat[0]/norm)
    part2 = inv_norm * arccos * quat[1:]

    logQuat = np.append(log_norm, part2)

    return logQuat


def generate_data(string, n_steps):
    """
    Create the dataset of data_type
    """
    geom_name = 'object_geom'

    model = mujoco_py.load_model_from_xml(string)
    sim = mujoco_py.MjSim(model)

    object_id = model.geom_names.index(geom_name)
    xyz_local = get_vert_local(sim, object_id)
    # viewer = mujoco_py.MjViewer(sim)

    dataset = {"pos": np.empty((n_steps//10, 8, 3)),
                "eucl_motion" : np.empty((n_steps//10, 1, 12)),
                "quat": np.empty((n_steps//10, 1, 7)),
                "log_quat": np.empty((n_steps//10, 1, 7)),
                "pos_diff": np.empty((n_steps//10, 8, 3)),
                "pos_diff_start": np.empty((n_steps//10, 8, 3))
              }

    for i in range(n_steps):
        sim.step()
        if i == 0:
            prev = get_vert_coords(sim, object_id-1, xyz_local).T
            start = prev
            dataset["pos_diff"][i] = np.zeros((8,3))
            dataset["pos_diff_start"][i] = np.zeros((8,3))
        if i % 10 == 0:
            dataset["pos"][i//10] = get_vert_coords(sim, object_id-1, xyz_local).T
            dataset["eucl_motion"][i//10] = np.append(get_mat(sim, object_id-1), sim.data.body_xpos[object_id-1])
            dataset["quat"][i//10] = np.append(get_quat(sim, object_id-1), sim.data.body_xpos[object_id-1])
            dataset["log_quat"][i//10] = np.append(calculate_log_quat(get_quat(sim, object_id-1)), sim.data.body_xpos[object_id-1])
            if i != 0:
                dataset["pos_diff"][i//10] = get_vert_coords(sim, object_id-1, xyz_local).T - prev
                prev = get_vert_coords(sim, object_id-1, xyz_local).T
                dataset["pos_diff_start"][i//10] = get_vert_coords(sim, object_id-1, xyz_local).T - start
    return dataset


def write_data_nsim(num_sims, n_steps, obj_type):
    for sim_id in range(num_sims):
        euler = f"{np.random.uniform(-80, 80)} {np.random.uniform(-80, 80)} {np.random.uniform(-80, 80)}"
        pos = f"{np.random.uniform(-100, 100)} {np.random.uniform(-100, 100)} {np.random.uniform(40, 300)}"
        size = f"{np.random.uniform(0.1, 1)} {np.random.uniform(0.1, 1)} {np.random.uniform(0.1, 1)}"

        string = create_string(euler, pos, obj_type, size)
        dataset = generate_data(string, n_steps)

        sim_data = {"vars" : [euler, pos, obj_type, size], "data" : dataset}
        with open(f'data/sim_{sim_id}.pickle', 'wb') as f:
            pickle.dump(sim_data, f)
        f.close()

obj_type = "box"
write_data_nsim(500, 30, obj_type)