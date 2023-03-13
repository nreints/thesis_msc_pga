import mujoco
import numpy as np
import itertools
import pickle
from convert import *
from pyquaternion import Quaternion
import os
import argparse
import time
import copy
import math
import random


def get_mat(data, obj_id):
    """
    Returns the rotation matrix of an object.

    Input:
        - data; MjData object containing the simulation information.
        - obj_id; id of the object.

    Output:
        - rotation matrix describing the motion.
    """
    return data.geom_xmat[obj_id].reshape(3, 3)


def get_vert_local(model, obj_id):
    """
    Returns the locations of the vertices centered around zero.

    Input:
        - model; MjModel object containing the simulation information.
        - obj_id; id of the object.

    Output:
        - xyz-positions of the vertices centered around zero and before rotation.
    """
    obj_size = model.geom_size[obj_id]
    offsets = np.array([-1, 1]) * obj_size[:, None]
    return np.stack(list(itertools.product(*offsets))).T


def get_vert_coords(data, obj_id, xyz_local):
    """
    Returns the locations of the vertices during simulation.

    Input:
        - data; MjData object containing the simulation information.
        - obj_id; id of the object.
        - xyz_local; xyz-positions of the vertices centered around zero and before rotation.

    Output:
        - xyz-coordinates of the vertices in the world frame.
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

    Input:
        - data; MjData object containing the simulation information.
        - obj_id; id of the object.

    Output:
        - Quaternion that describes the rotation of the object with obj_id.
            a bi cj dk convention (identity: 1 0 0 0)
    """
    quat = data.xquat[obj_id]
    # Ensure that the first element of the quaternion is positive. TODO Steven van Leo
    # if quat[0] < 0:
    #     quat *= -1
    return quat


def calculate_log_quat(quat):
    """
    Calculates the log quaternion based on the quaternion.

    Input:
        - quat; quaternion that describes the rotation (4 dimensional). Convention [a, bi, cj, dk].

    Output:
        - Logarithm of the quaternion. Calculated according
        to https://en.wikipedia.org/wiki/Quaternion#Exponential,_logarithm,_and_power_functions.
    """
    norm = np.linalg.norm(quat)
    log_norm = np.log(norm)

    if np.linalg.norm(quat[1:]) == 0:
        inv_norm = 0
    else:
        inv_norm = 1 / np.linalg.norm(quat[1:])

    part_arccos = np.arccos(quat[0] / norm)
    part2 = inv_norm * part_arccos * quat[1:]

    logQuat = np.append(log_norm, part2)
    return logQuat


def get_dualQ(quat, translation):
    """
    Returns the dualquaternion of an object.

    Input:
        - quat; quaternion that describes the rotation (4 dimensional). Convention [a, bi, cj, dk].
        - translation; translation vector that describes the translation (3 dimensional).

    Output:
        - Dual Quaternion that describes the rotation and translation (8 dimensional).
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
    Returns the logarithm of a dual quaternion r.

    Input:
        - r: rotor / dual quaternion (8 dimensional).

    Output:
        - bivector / logarithm of a rotor (6 dimensional).
    (14 mul, 5 add, 1 div, 1 acos, 1 sqrt)
    """

    if r[0] == 1 or math.isclose(r[0], 1):
        return np.array([-r[5], -r[6], -r[7], 0, 0, 0])

    a = 1 / (1 - r[0] * r[0])

    b = np.arccos(r[0]) * np.sqrt(a)
    c = a * r[4] * (1 - r[0] * b)
    return np.array(
        [
            c * r[1] - b * r[5],
            c * r[2] - b * r[6],
            c * r[3] - b * r[7],
            b * r[3],
            b * r[2],
            b * r[1],
        ]
    )


def create_empty_dataset(n_steps, half_size, mass):
    """
    Returns empty data dictionary.

    Input:
        - n_steps; number of steps in simulation.

    Output:
        - Dictionary to store the data in.
    """
    # print(half_size)
    size = half_size * 2
    size_squared = size**2
    # print(size)
    # print(size_squared)
    # exit()
    return {
        "pos": np.empty((n_steps, 8, 3)),
        "eucl_motion": np.empty((n_steps, 1, 12)),
        "quat": np.empty((n_steps, 1, 7)),
        "log_quat": np.empty((n_steps, 1, 7)),
        "dual_quat": np.empty((n_steps, 1, 8)),
        "pos_diff_start": np.empty((n_steps, 8, 3)),
        "log_dualQ": np.empty((n_steps, 6)),
        "rotation_axis_trans": np.empty((n_steps, 6)),
        "inertia_body": np.empty((3, 1)),
        "size": size,
        "size_squared": size_squared,
        "size_mass": np.append(size, mass),
        "size_squared_mass": np.append(size_squared, mass),
        "size_centroid": np.empty((2, 3)),
        "size_squared_centroid": np.empty((2, 3)),
        "size_massCentroid": np.empty((2, 3)),
        "size_squared_massCentroid": np.empty((2, 3)),
    }


def generate_data(
    string,
    n_steps,
    visualize=False,
    vel_range_l=(0, 0),
    vel_range_a=(0, 0),
    pure_tennis=False,
):
    """
    Create the dataset (dictionary) of data_type for n_steps steps.

    Input:
        - string; XML string of the model.
        - n_steps; number of steps to generate.
        - visualize; boolean;
            - True; visualize in MuJoCo.
            - False; do not visualize in MuJoCo.
        - vel_range_l; range to choose values from for the linear velocity.
        - vel_range_a; range to choose values from for the angular velocity.

    Output:
        - dataset; dictionary with all data.
    """

    # Generate model object.
    model = mujoco.MjModel.from_xml_string(string)

    # Generate MjData object
    data = mujoco.MjData(model)

    # print("body_inertia", model.body_inertia[1])
    # print("meanmass", model.stat.meanmass)
    # H = np.zeros((model.nv, model.nv))
    # print("qM", data.qM)
    # print("qLD", data.qLD)
    # L = mujoco.mj_fullM(model, H, data.qM)
    # # print("L", L)
    # print("nM, nv", model.nM, model.nv)

    # Set linear (qvel[0:3]) and angular (qvel[3:6]) velocity
    data.qvel[0:3] = np.random.uniform(vel_range_l[0], vel_range_l[1], size=3)
    # data.qvel[0:3] = [0, -3, 0]
    data.qvel[3:6] = np.random.uniform(vel_range_a[0], vel_range_a[1], size=3)
    if pure_tennis:
        data.qvel[3:6] = [0, random.uniform(15, 40), 0.01]

    # Collect geom_id and body_id
    geom_id = model.geom("object_geom").id
    body_id = model.body("object_body").id

    # Calculate vertice positions before rotation and translation.
    xyz_local = get_vert_local(model, geom_id)

    # Initialize data dictionary
    dataset = create_empty_dataset(
        n_steps, model.geom_size[geom_id], model.stat.meanmass
    )

    if visualize:
        import mujoco_viewer

        viewer = mujoco_viewer.MujocoViewer(model, data)

    for i in range(0, n_steps):
        if not visualize or viewer.is_alive:
            mujoco.mj_step(model, data)

            if visualize:
                viewer.render()

            xpos = data.geom_xpos[geom_id]
            global_pos = get_vert_coords(data, geom_id, xyz_local).T

            # Collect position data after rotation and translation.
            dataset["pos"][i] = global_pos

            # print("--------")
            # H = np.zeros((model.nv, model.nv))
            # print("body_inertia", model.body_inertia)
            # print("qLD", data.qLD)  # number of non-zeros in sparse inertia matrix
            # print(
            #     "diag(D)", 1 / data.qLDiagInv
            # )  # number of degrees of freedom = dim(qvel)
            # print(
            #     "sqrt(diag(D))", 1 / data.qLDiagSqrtInv
            # )  # number of degrees of freedom = dim(qvel)

            # print("qM", data.qM)
            # L = mujoco.mj_fullM(model, H, data.qM)
            # # print("L", L)
            # print("nM, nv", model.nM, model.nv)
            # print(data.qM == data.qLD)
            # # if i == 3:
            # #     exit()

            if i == 0:
                start_xpos = copy.deepcopy(xpos)

                start_xyz = global_pos

                # First difference should be zero
                dataset["pos_diff_start"][i] = np.zeros((8, 3))

                start_rotMat = copy.deepcopy(get_mat(data, geom_id))
                dataset["eucl_motion"][i] = np.append(np.eye(3), np.zeros(3))

                start_quat = copy.deepcopy(get_quat(data, body_id))
                prev_quat = start_quat
                dataset["quat"][i] = np.append([1, 0, 0, 0], np.zeros(3))
                dataset["log_quat"][i] = np.append([0, 0, 0, 0], np.zeros(3))

                dualQ_start = get_dualQ([1, 0, 0, 0], np.zeros(3))
                dataset["dual_quat"][i] = dualQ_start
                dataset["log_dualQ"][i] = logDual(dualQ_start)

                rotation_axis = Quaternion([1, 0, 0, 0]).axis
                dataset["rotation_axis_trans"][i] = np.append(rotation_axis, xpos)

            else:
                # Collect rotation matrix
                current_rotMat = get_mat(data, geom_id)

                rel_trans = (
                    xpos - current_rotMat @ np.linalg.inv(start_rotMat) @ start_xpos
                )
                # print(rel_trans)
                rel_rot = current_rotMat @ np.linalg.inv(start_rotMat)

                dataset["eucl_motion"][i] = np.append(rel_rot.flatten(), rel_trans)

                quaternion_pyquat = (
                    Quaternion(get_quat(data, body_id)) * Quaternion(start_quat).inverse
                )
                # print(quaternion_pyquat.elements, quaternion_pyquat.axis)
                # TODO Steven van Leo
                # if quaternion_pyquat.elements[0] < 0:
                #     quaternion_pyquat *= -1
                # print(quaternion_pyquat.elements, quaternion_pyquat.axis)
                # print("----")
                rotation_axis = quaternion_pyquat.axis

                dataset["rotation_axis_trans"][i] = np.append(rotation_axis, xpos)

                quaternion = quaternion_pyquat.elements
                dataset["quat"][i] = np.append(quaternion, rel_trans)
                # Collect Log Quaternion data
                dataset["log_quat"][i] = np.append(
                    calculate_log_quat(quaternion), rel_trans
                )
                # print(calculate_log_quat(quaternion))

                dualQuaternion = get_dualQ(quaternion, rel_trans)
                # print(dualQuaternion)
                # Collect Dual-Quaternion data
                dataset["dual_quat"][i] = dualQuaternion

                # Collect log_dualQ data (= bivector = rotation axis)
                dataset["log_dualQ"][i] = logDual(dualQuaternion)

                dataset["pos_diff_start"][i] = (
                    get_vert_coords(data, geom_id, xyz_local).T - start_xyz
                )
            # if i>100:
            #     exit()
        else:
            print("Visualisation failed")
            break

    if visualize:
        viewer.close()

    return dataset


def get_sizes(symmetry):
    """
    Returns the sizes given the required symmetry.

    Input:
        - symmetry; symmetry type of the box
            - full; ratio of the vertices 1:1:1
            - semi; ratio of the vertices 1:1:10
            - tennis; ratio of the vertices 1:3:10
            - none; no specific ratio

    Output:
        - String containing the lengths of hight, width, and depth.
    """
    if symmetry == "none":
        return f"{np.random.uniform(0.5, 5)} {np.random.uniform(0.5, 5)} {np.random.uniform(0.5, 5)}"
    elif symmetry == "full":
        ratio = np.array([1, 1, 1])
    elif symmetry == "semi":
        ratio = np.array([1, 1, 10])
    elif symmetry == "tennis":
        ratio = np.array([1, 3, 10])
    else:
        raise argparse.ArgumentError(
            f"Not a valid string for argument symmetry: {symmetry}"
        )
    random_size = np.random.uniform(
        0.5, 5
    )  # TODO willen we dat ze gemiddeld even groot zijn? Ik heb nu dat de kortste zijde gemiddeld even groot is.
    sizes = ratio * random_size
    # print("sizes:", sizes)
    # return "5 25 10", [5, 25, 10]
    return f"{sizes[0]} {sizes[1]} {sizes[2]}", sizes


def get_dir(vel_range_l, vel_range_a, symmetry, num_sims, plane, grav, tennis_effect):
    """
    Returns the name of the directory to write to.

    Input:
        - vel_range_l; range of initial linear velocity.
        - vel_range_a; range of initial angular velocity.
        - symmetry; shape of cuboid.
        - num_sims; number of sims to generate.
        - plane; boolean whether there is a plane in the simulation.
        - grav; boolean whether there is gravity in the simulation.

    Output:
        - Directory with corresponding name.
    """
    dir = f"data/data_t{vel_range_l}_r{vel_range_a}_{symmetry}_p{plane}_g{grav}"
    if tennis_effect:
        dir = f"data/data_t{vel_range_l}_r{vel_range_a}_{symmetry}_p{plane}_g{grav}_tennisEffect"
    if not os.path.exists("data"):
        os.mkdir("data")
    if not os.path.exists(dir):
        print("Creating directory")
        os.mkdir(dir)
    # Warn if directory already exists with more simulations.
    elif len(os.listdir(dir)) > num_sims:
        print(
            f"This directory already existed with {len(os.listdir(dir))} files, you want {num_sims} files. Please delete directory manually."
        )
        raise IndexError(
            f"This directory ({dir}) already exists with fewer simulations."
        )
    return dir


def get_string(euler_obj, pos_obj, size_obj, gravity, plane, integrator):
    """
    Creates the XML string for a simulation.

    Input:
        - euler_obj; euler orientation of the object.
        - pos_obj; xyz-position of the objects center.
        - size_obj; size of the object.
        - gravity; boolean;
            - True; use gravity in the simulation.
            - False; use no gravity in the simulation.
        - plane; boolean;
            - True; create a plane.
            - False; create no plane.
        - integrator; type of integrator to use in MuJoCo.

    Output:
        - XML string to create a MuJoCo simulation.
    """
    if plane:
        plane_str = '<geom type="plane" pos="0 0 0" size="10 10 10" rgba="1 1 1 1"/>'
    else:
        plane_str = ""

    if gravity:
        gravity_str = f'<option integrator="{integrator}">'
    else:
        gravity_str = (
            f'<option integrator="{integrator}" gravity="0 0 0" iterations="10"/>'
        )
    # size = size_obj.split(" ")
    # print(f"sizes {size}")
    # product = 1000
    # for el in size:
    #     product *= float(el) * 2
    # I_xx = 1 / 12 * product * ((float(size[1]) * 2) ** 2 + (2 * float(size[2])) ** 2)
    # I_yy = 1 / 12 * product * ((float(size[2]) * 2) ** 2 + (2 * float(size[0])) ** 2)
    # I_zz = 1 / 12 * product * ((float(size[1]) * 2) ** 2 + (2 * float(size[0])) ** 2)
    # print("own Ixx, Iyy, Izz", I_xx, I_yy, I_zz)
    return f"""
    <mujoco>
    {gravity_str}
    <worldbody>
        <light name="top" pos="0 0 1"/>
        <camera name="camera1" pos="1 -70 50" xyaxes="1 0 0 0 1 1.5"/>
        <camera name="camera2" pos="10 -70 70" xyaxes="1 0 0 0 1 1.5"/>
        <body name="object_body" euler="{euler_obj}" pos="{pos_obj}">
            <joint name="joint1" type="free"/>
            <geom name="object_geom" type="box" size="{size_obj}" rgba="1 0 0 1"/>
        </body>
        {plane_str}
    </worldbody>
    </mujoco>
    """


def write_data_nsim(
    num_sims,
    n_steps,
    symmetry,
    gravity,
    dir,
    visualize,
    vel_range_l,
    vel_range_a,
    plane,
    integrator,
    tennis_effect,
):
    """
    Computes and writes data of num_sims each with n_steps.

    Input:
        - num_sims; number of simulations.
        - n_steps; number of steps per simulation.
        - symmetry;
        - gravity; boolean;
            - True; use gravity in the simulation.
            - False; use no gravity in the simulation.
        - dir; data directory to save the pickle files in.
        - visualize;  boolean;
            - True; visualize in MuJoCo.
            - False; do not visualize in MuJoCo.
        - vel_range_l; range to choose values from for the linear velocity.
        - vel_range_a; range to choose values from for the angular velocity.
        - plane; boolean;
            - True; create a plane.
            - False; create no plane.
        - integrator; type of integrator to use in MuJoCo.
        - tennis_effect; explicitly cause the tennis effect.

    Output:
        - None; writes to the corresponding pickle file.
    """
    if tennis_effect and not symmetry == "tennis":
        raise BaseException("Cannot create tenniseffect if symmetry is not tennis.")
    for sim_id in range(num_sims):
        if sim_id % 100 == 0 or sim_id == num_sims - 1:
            print(f"Generating sim {sim_id}/{num_sims-1}")
        # Define euler angle
        euler = f"{np.random.uniform(0, 360)} {np.random.uniform(0, 360)} {np.random.uniform(0, 360)}"
        # euler = "0 0 0"
        # Define sizes
        sizes_str, sizes_list = get_sizes(symmetry)
        # Define position
        pos = f"{np.random.uniform(-10, 10)} {np.random.uniform(-10, 10)} {np.random.uniform(-10, 10)}"
        # pos = "0 0 0"
        string = get_string(euler, pos, sizes_str, gravity, plane, integrator)
        # Create dataset
        dataset = generate_data(
            string, n_steps, visualize, vel_range_l, vel_range_a, tennis_effect
        )
        sim_data = {
            "vars": {
                "euler": euler,
                "pos": pos,
                "sizes": sizes_list,
                "gravity": gravity,
                "n_steps": n_steps,
            },
            "data": dataset,
        }
        # Write data to file
        with open(f"{dir}/sim_{sim_id}.pickle", "wb") as f:
            pickle.dump(sim_data, f)


if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("-n_sims", type=int, help="number of simulations", default=5000)
    parser.add_argument("-n_frames", type=int, help="number of frames", default=1000)
    parser.add_argument(
        "-s",
        "--symmetry",
        type=str,
        choices=["full", "semi", "tennis", "none"],
        help="symmetry of the box.\nfull: symmetric box 1:1:1\n; semi: 2 sides of same length, other longer 1:1:10\n;tennis: tennis_racket effect 1:3:10\n;none: random lengths for each side",
        default="tennis",
    )
    parser.add_argument("-l_min", type=int, help="linear qvel min", default=5)
    parser.add_argument("-l_max", type=int, help="linear qvel max", default=8)
    parser.add_argument("-a_min", type=int, help="angular qvel min", default=0)
    parser.add_argument("-a_max", type=int, help="angular qvel max", default=0)
    parser.add_argument(
        "-integrator",
        type=str,
        choices=["RK4", "Euler"],
        help="type of integrator to use",
        default="Euler",
    )
    parser.add_argument("--gravity", action=argparse.BooleanOptionalAction)
    parser.add_argument("--plane", action=argparse.BooleanOptionalAction)
    parser.add_argument("--visualize", action=argparse.BooleanOptionalAction)
    parser.add_argument("--tennis_effect", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    vel_range_l = (args.l_min, args.l_max)
    vel_range_a = (args.a_min, args.a_max)

    print(
        f"Creating dataset vel_range_l={vel_range_l}, vel_range_a={vel_range_a}, symmetry={args.symmetry}"
    )

    data_dir = get_dir(
        vel_range_l,
        vel_range_a,
        args.symmetry,
        args.n_sims,
        args.plane,
        args.gravity,
        args.tennis_effect,
    )

    write_data_nsim(
        args.n_sims,
        args.n_frames,
        args.symmetry,
        args.gravity,
        data_dir,
        args.visualize,
        vel_range_l,
        vel_range_a,
        args.plane,
        args.integrator,
        args.tennis_effect,
    )
    print(f"Saved in {data_dir}")
    print(f"\nTime: {time.time()- start_time}\n---- FINISHED ----")
