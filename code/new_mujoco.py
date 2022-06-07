import mujoco_py
import numpy as np
import itertools
from create_strings import create_string
import pickle

VERT_NUM = 1

def rotVecQuat(v, q):
    # From internet
    res = np.zeros(3)
    mujoco_py.functions.mju_rotVecQuat(res, v, q)
    return res

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
    rot_vec = quat[1:] / np.linalg.norm(quat[1:])
    sin_vec = quat[1:] / rot_vec
    sin = sin_vec[0]
    cos = quat[0]
    angle = 2 * np.arctan(cos/sin)
    return np.append(rot_vec, angle)

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
                "log_quat": np.empty((n_steps//10, 1, 7))}

    for i in range(n_steps):
        sim.step()
        if i% 10 == 0:
            dataset["pos"][i//10] = get_vert_coords(sim, object_id-1, xyz_local).T
            dataset["eucl_motion"][i//10] = np.append(get_mat(sim, object_id-1), sim.data.body_xpos[object_id-1])
            dataset["quat"][i//10] = np.append(get_quat(sim, object_id-1), sim.data.body_xpos[object_id-1])
            dataset["log_quat"][i//10] = np.append(calculate_log_quat(get_quat(sim, object_id-1)), sim.data.body_xpos[object_id-1])

    return dataset


def write_data_nsim(num_sims, n_steps):
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

if __name__ == "__main__":

    obj_type = "box"
    n_steps = 400

    num_sims = 500
    write_data_nsim(num_sims, n_steps)

    # with open(f'data/eucl_motion/sim_0.pickle', 'rb') as f:
    #     print(np.shape(pickle.load(f)["data"].flatten()))


    """
    Vragen:

    """