import mujoco_py
import numpy as np
import itertools
import xml_strings
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
    # Geeft nu alleen de geroteerde eerste vertice terug [:,??]
    return rotVecQuat(xyz_local[:,VERT_NUM], obj_xquat) + sim.data.body_xpos[obj_id]

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

def generate_data(string, n_steps, data_type="pos", dim=3):
    """
    Create the dataset of data_type
    """
    geom_name = 'object_geom'

    model = mujoco_py.load_model_from_xml(string)
    sim = mujoco_py.MjSim(model)

    object_id = model.geom_names.index(geom_name)
    xyz_local = get_vert_local(sim, object_id)
    # viewer = mujoco_py.MjViewer(sim)

    # print("quat", get_vert_coords_quat(sim, object_id-1, xyz_local))
    # print("old ", get_vert_coords(sim, object_id-1, xyz_local)[:,VERT_NUM])

# -----------
    # dataset = np.empty((n_steps*8, dim if data_type=="pos" else 4))
    # for i in range(n_steps):
    #     sim.step()
    #     if data_type == "pos":
    #         dataset[i*8:((i+1)*8)] = get_vert_coords(sim, object_id-1, xyz_local).T
    #     elif data_type == "quat": #fix
    #         dataset[i] = get_vert_coords_quat(sim, object_id-1, xyz_local)


    dataset = np.empty((n_steps, 8, dim if data_type=="pos" else 4))

    for i in range(n_steps):
        sim.step()
        print(get_vert_coords(sim, object_id-1, xyz_local).T.shape)
        if data_type == "pos":
            dataset[i] = get_vert_coords(sim, object_id-1, xyz_local).T

        # viewer.render()

    # print("quat", get_vert_coords_quat(sim, object_id-1, xyz_local))
    # print("old ", get_vert_coords(sim, object_id-1, xyz_local)[:,VERT_NUM])

    return dataset


def write_data_nsim(num_sims):
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
    # string = xml_strings.model_xml_swinging_body
    # euler = "0 0 -80"
    # pos = "0 0 100"
    # size = "0.2 0.2 0.2"

    obj_type = "box"
    n_steps = 400

    num_sims = 100
    write_data_nsim(num_sims)

    # with open(f'data/sim_0.pickle', 'rb') as f:
    #     print(np.shape(pickle.load(f)["data"].flatten()))




    """
    Vragen:

    """