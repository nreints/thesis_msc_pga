import mujoco_py
import numpy as np
import itertools
from create_strings import create_string
import pickle
import torch
# from pyquaternion import Quaternion
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
    """
    Returns the rotated batch of vectors v by a batch quaternion q.
    v shape: batchx8x3
    q shape: batchx4
    """
    device = v.device

    v_reshaped = v.reshape((v.shape[0]*v.shape[1], -1))


    # Swap columns for roma calculations (bi, cj, dk, a)
    q_new = torch.index_select(q, 1, torch.tensor([1, 2, 3, 0]).to(device))
    # print("q swapped", q_new.shape, "\n", q_new[80])

    q_new = torch.repeat_interleave(q_new, repeats=8, dim=0)
    # print("q repeated", q_new.shape, "\n", q_new[640:660])
    v_new = torch.hstack((v_reshaped, torch.zeros(v_reshaped.shape[0],1).to(device)))

    # Calculate q* v q
    mult = roma.quat_product(v_new, q_new)
    q_conj = roma.quat_conjugation(q_new)
    mult2 = roma.quat_product(q_conj, mult)

    # Remove zeros, reshape to v shape
    rotated_vec = mult2[:,:-1].reshape(v.shape[0], v.shape[1], -1)


    return rotated_vec

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


# def rotVecQuat(v, q):
#     # From internet MuJoCo
#     res = np.zeros(3)
#     mujoco_py.functions.mju_rotVecQuat(res, v, q)
#     return res


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

    # TODO np.linalg.norm(quat[1:]) = 0
    if np.linalg.norm(quat[1:]) == 0:
        inv_norm = 0
    else:
        inv_norm = 1 / np.linalg.norm(quat[1:])


    arccos = np.arccos(quat[0]/norm)
    part2 = inv_norm * arccos * quat[1:]

    logQuat = np.append(log_norm, part2)

    return logQuat


def generate_data(string, n_steps, visualize):
    """
    Create the dataset of data_type
    """
    geom_name = 'object_geom'

    model = mujoco_py.load_model_from_xml(string)
    sim = mujoco_py.MjSim(model)

    object_id = model.geom_names.index(geom_name)
    xyz_local = get_vert_local(sim, object_id)
    if visualize:
        viewer = mujoco_py.MjViewer(sim)

    dataset = {"pos": np.empty((n_steps//10, 8, 3)),
                "eucl_motion" : np.empty((n_steps//10, 1, 12)),
                "quat": np.empty((n_steps//10, 1, 7)),
                "log_quat": np.empty((n_steps//10, 1, 7)),
                "pos_diff": np.empty((n_steps//10, 8, 3)),
                "pos_diff_start": np.empty((n_steps//10, 8, 3)),
                "pos_norm": np.empty((n_steps//10, 8, 3))
              }

    for i in range(n_steps):
        sim.step()
        if visualize:
            viewer.render()
        if i == 0:
            prev = get_vert_coords(sim, object_id-1, xyz_local).T
            start = prev
            # First quaternion should be a identity quaternion with no translation
            dataset["quat"][i] = np.array([1, 0, 0, 0, 0, 0, 0])
            # TODO identity log quaternion
            dataset["log_quat"][i] = np.append(calculate_log_quat(np.array([1, 0, 0, 0])), np.zeros(3))
            # First euclidean motion is identity rotation and no translation
            dataset["eucl_motion"][i] = np.append(np.eye(3).flatten(), np.zeros((3)))
            # First difference should be zero
            dataset["pos_diff"][i] = np.zeros((8, 3))
            dataset["pos_diff_start"][i] = np.zeros((8, 3))
            # First position should be the position
            dataset["pos"][i//10] = get_vert_coords(sim, object_id-1, xyz_local).T
        if i % 10 == 0 and i != 0:
            dataset["pos"][i//10] = get_vert_coords(sim, object_id-1, xyz_local).T
            dataset["eucl_motion"][i//10] = np.append(get_mat(sim, object_id-1), sim.data.body_xpos[object_id-1])
            dataset["quat"][i//10] = np.append(get_quat(sim, object_id-1), sim.data.body_xpos[object_id-1])
            dataset["log_quat"][i//10] = np.append(calculate_log_quat(get_quat(sim, object_id-1)), sim.data.body_xpos[object_id-1])
            if i != 0:
                dataset["pos_diff"][i//10] = get_vert_coords(sim, object_id-1, xyz_local).T - prev
                prev = get_vert_coords(sim, object_id-1, xyz_local).T
                dataset["pos_diff_start"][i//10] = get_vert_coords(sim, object_id-1, xyz_local).T - start

    print(len(dataset["quat"]) == len(dataset["pos"]))
    dataset["pos_norm"] = (dataset["pos"] - np.mean(dataset["pos"], axis=(0,1))) / np.std(dataset["pos"], axis=(0,1))
    # print(np.mean(dataset["pos"], axis=(0,1)), np.std(dataset["pos"], axis=(0,1)))
    return dataset


def write_data_nsim(num_sims, n_steps, obj_type, visualize=False):
    for sim_id in range(num_sims):
        print("sim: ", sim_id)
        euler = f"{np.random.uniform(-80, 80)} {np.random.uniform(-80, 80)} {np.random.uniform(-80, 80)}"
        pos = f"{np.random.uniform(-10, 10)} {np.random.uniform(-10, 10)} {np.random.uniform(4, 30)}"
        size = f"{np.random.uniform(0.1, 1)} {np.random.uniform(0.1, 1)} {np.random.uniform(0.1, 1)}"

        string = create_string(euler, pos, obj_type, size)
        dataset = generate_data(string, n_steps, visualize)

        sim_data = {"vars" : [euler, pos, obj_type, size], "data" : dataset}
        with open(f'data/sim_{sim_id}.pickle', 'wb') as f:
            pickle.dump(sim_data, f)
        f.close()

if __name__ == "__main__":
    ## Uncomment to create random data
    n_sims = 750
    n_steps = 2250
    obj_type = "box"

    write_data_nsim(n_sims, n_steps, obj_type, visualize=False)