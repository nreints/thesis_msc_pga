import argparse
import copy
import itertools
import math
import os
import pickle
import random
import time

import mujoco
import numpy as np
from pyquaternion import Quaternion

from convert import *


def get_mat(data, obj_id):
    """
    Returns the rotation matrix of an object.

    Input:
        - data; MjData object containing the simulation information.
        - obj_id; id of the object.

    Output:
        - rotation matrix describing the motion.
            Shape: (3, 3).
    """
    return data.geom_xmat[obj_id].reshape(3, 3)


def get_vert_local(model, obj_id):
    """
    Returns the locations of the vertices centered around zero before rotation.

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


def dualQ_normalize(dualQ):
    A = 1 / np.sqrt(dualQ[0] ** 2 + dualQ[3] ** 2 + dualQ[2] ** 2 + dualQ[1] ** 2)
    B = (
        (
            dualQ[4] * dualQ[0]
            - (-dualQ[5] * dualQ[1] - dualQ[6] * dualQ[2] - dualQ[7] * dualQ[3])
        )
        * A
        * A
        * A
    )
    res = np.zeros_like(dualQ)
    res[0] = A * dualQ[0]
    res[5] = -(A * -dualQ[5] + B * dualQ[1])
    res[6] = -(A * -dualQ[6] + B * dualQ[2])
    res[7] = -(A * -dualQ[7] + B * dualQ[3])
    res[3] = A * dualQ[3]
    res[2] = A * dualQ[2]
    res[1] = A * dualQ[1]
    res[4] = A * dualQ[4] - B * dualQ[0]
    return res


def get_dualQ(quat, translation):
    """
    Returns the dualquaternion of an object.

    Input:
        - quat; quaternion that describes the rotation. Convention [a, bi, cj, dk].
            Shape (4,1)
        - translation; translation vector that describes the translation.
            Shape (3,1)

    Output:
        - Dual Quaternion that describes the rotation and translation.
            Shape (8,1)
    """
    # https://cs.gmu.edu/~jmlien/teaching/cs451/uploads/Main/dual-quaternion.pdf
    qr = quat
    t = np.hstack((np.array([0]), translation))

    qd = (0.5 * Quaternion(t) * Quaternion(quat)).elements

    dualQ = np.append(qr, qd)
    return dualQ_normalize(dualQ)


def logDual(r):
    """
    Returns the logarithm of a dual quaternion r.

    Input:
        - r: rotor / dual quaternion.
            - Shape (8,1)

    Output:
        - bivector / logarithm of a rotor (6 dimensional).
            - Shape (6,1)
    (14 mul, 5 add, 1 div, 1 acos, 1 sqrt)
    """

    if r[0] == 1 or math.isclose(r[0], 1):
        return np.array([-r[5], -r[6], -r[7], 0, 0, 0])

    a = 1 / (1 - r[0] * r[0])  # inv squared length

    b = np.arccos(r[0]) * np.sqrt(a)  # rotation scale
    c = a * r[4] * (1 - r[0] * b)  # translation scale
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


def create_empty_dataset(n_steps, half_size, mass, body_inertia):
    """
    Returns empty data dictionary.

    Input:
        - n_steps; number of steps in simulation.
        - half_size; half of the length, width, and height of the object.
        - mass; mass of the object
        - body_inertia; #TODO principal moments of inertia of the object.

    Output:
        - Dictionary to store the data in.
    """
    size = half_size * 2
    size_squared = size**2
    return {
        "pos": np.empty((n_steps, 8, 3)),
        "rot_mat": np.empty((n_steps, 1, 12)),
        "quat": np.empty((n_steps, 1, 7)),
        "log_quat": np.empty((n_steps, 1, 7)),
        "dual_quat": np.empty((n_steps, 1, 8)),
        "log_dualQ": np.empty((n_steps, 6)),
        "pos_diff_start": np.empty((n_steps, 8, 3)),
        "rot_mat_1": np.empty((n_steps, 1, 12)),
        "quat_1": np.empty((n_steps, 1, 7)),
        "log_quat_1": np.empty((n_steps, 1, 7)),
        "dual_quat_1": np.empty((n_steps, 1, 8)),
        "log_dualQ_1": np.empty((n_steps, 6)),
        "pos_diff_1": np.empty((n_steps, 8, 3)),
        # "rot_mat_ori": np.empty((n_steps, 1, 12)),
        # "quat_ori": np.empty((n_steps, 1, 7)),
        # "log_quat_ori": np.empty((n_steps, 1, 7)),
        # "dual_quat_ori": np.empty((n_steps, 1, 8)),
        # "log_dualQ_ori": np.empty((n_steps, 6)),
        "rotation_axis_trans": np.empty((n_steps, 6)),
        "inertia_body": body_inertia,
        "size": size,
        "size_squared": size_squared,
        "size_mass": np.append(size, mass),
        "size_squared_mass": np.append(size_squared, mass),
        # "size_centroid": np.empty((2, 3)),
        # "size_squared_centroid": np.empty((2, 3)),
        # "size_massCentroid": np.empty((n_steps, 2, 3)),
        # "size_squared_massCentroid": np.empty((n_steps, 2, 3)),
        "start": np.empty((8, 3)),
        "xpos_start": np.empty((1, 3)),
        "xpos": np.empty((n_steps, 3)),
    }


def generate_data(
    string,
    n_steps,
    dict_name,
    grav_coll,
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
        - pure_tennis; boolean
            True; ensure the tennis effect occurs
            False; the tennis effect might occur

    Output:
        - dataset; dictionary with all data.
    """

    # Generate model object.
    model = mujoco.MjModel.from_xml_string(string)

    # Generate MjData object
    data = mujoco.MjData(model)

    # Generate random unit vector
    u1 = np.random.uniform(-1, 1, size=3)
    u1_normalized = u1 / np.linalg.norm(u1)

    if grav_coll:
        u1_normalized[-1] = -abs(u1_normalized[-1])

    u2 = np.random.uniform(-1, 1, size=3)
    u2_normalized = u2 / np.linalg.norm(u2)

    # Set linear (qvel[0:3]) and angular (qvel[3:6]) velocity
    data.qvel[0:3] = u1_normalized * np.random.uniform(vel_range_l[0], vel_range_l[1])
    data.qvel[3:6] = u2_normalized * np.random.uniform(vel_range_a[0], vel_range_a[1])

    if pure_tennis:
        data.qvel[3:6] = [0, random.uniform(30, 50), 1]

    # Collect geom_id and body_id
    geom_id = model.geom("object_geom").id
    body_id = model.body("object_body").id

    # Calculate vertice positions before rotation and translation.
    xyz_local = get_vert_local(model, geom_id)

    # Initialize data dictionary
    dataset = create_empty_dataset(
        n_steps,
        model.geom_size[geom_id],
        model.stat.meanmass,
        model.body_inertia[body_id],
    )
    dataset["start"] = xyz_local.T

    if visualize:
        import mujoco_viewer

        viewer = mujoco_viewer.MujocoViewer(model, data)

    for i in range(0, n_steps, 1):
        if not visualize or viewer.is_alive:
            mujoco.mj_step(model, data)

            if visualize:
                viewer.render()

            current_xpos = copy.deepcopy(data.geom_xpos[geom_id])
            dataset["xpos"][i] = current_xpos
            current_pos = copy.deepcopy(get_vert_coords(data, geom_id, xyz_local).T)
            current_rotMat = copy.deepcopy(get_mat(data, geom_id))

            # Collect position data after rotation and translation.
            dataset["pos"][i] = current_pos

            if i == 0:
                start_xpos = copy.deepcopy(current_xpos)
                prev_xpos = start_xpos
                dataset["xpos_start"] = start_xpos

                start_xyz = current_pos
                prev_pos = current_pos

                # First difference should be zero
                dataset["pos_diff_start"][i] = np.zeros((8, 3))
                dataset["pos_diff_1"][i] = np.zeros((8, 3))

                start_rotMat = copy.deepcopy(get_mat(data, geom_id))
                prev_rotMat = start_rotMat
                dataset["rot_mat"][i] = np.append(np.eye(3), np.zeros(3))
                dataset["rot_mat_1"][i] = np.append(np.eye(3), np.zeros(3))

                start_quat = copy.deepcopy(get_quat(data, body_id))
                prev_quat = start_quat
                dataset["quat"][i] = np.append([1, 0, 0, 0], np.zeros(3))
                dataset["quat_1"][i] = np.append([1, 0, 0, 0], np.zeros(3))
                dataset["log_quat"][i] = np.append([0, 0, 0, 0], np.zeros(3))
                dataset["log_quat_1"][i] = np.append([0, 0, 0, 0], np.zeros(3))

                dualQ_start = get_dualQ([1, 0, 0, 0], np.zeros(3))
                dataset["dual_quat"][i] = dualQ_start
                dataset["dual_quat_1"][i] = dualQ_start
                dataset["log_dualQ"][i] = logDual(dualQ_start)
                dataset["log_dualQ_1"][i] = logDual(dualQ_start)

                rotation_axis = Quaternion([1, 0, 0, 0]).axis
                dataset["rotation_axis_trans"][i] = np.append(rotation_axis, start_xpos)

            else:
                # Collect rotation matrix
                rel_trans = current_xpos - start_xpos
                rel_trans1 = current_xpos - prev_xpos
                prev_xpos = current_xpos
                rel_rot = current_rotMat @ np.linalg.inv(start_rotMat)
                rel_rot_1 = current_rotMat @ np.linalg.inv(prev_rotMat)
                prev_rotMat = current_rotMat
                dataset["rot_mat"][i][:, :9] = rel_rot.flatten()
                dataset["rot_mat"][i][:, 9:] = rel_trans
                dataset["rot_mat_1"][i][:, :9] = rel_rot_1.flatten()
                dataset["rot_mat_1"][i][:, 9:] = rel_trans1

                rel_quaternion_pyquat = (
                    Quaternion(get_quat(data, body_id)) * Quaternion(start_quat).inverse
                ).normalised
                rel_quaternion1_pyquat = (
                    Quaternion(get_quat(data, body_id)) * Quaternion(prev_quat).inverse
                ).normalised

                # TODO Steven van Leo
                # if quaternion_pyquat.elements[0] < 0:
                #     quaternion_pyquat *= -1
                # print(quaternion_pyquat.elements, quaternion_pyquat.axis)

                rel_quaternion = rel_quaternion_pyquat.elements
                rel_quaternion1 = rel_quaternion1_pyquat.elements
                prev_quat = copy.deepcopy(get_quat(data, body_id))

                rotation_axis = rel_quaternion_pyquat.axis
                dataset["rotation_axis_trans"][i][:3] = rotation_axis
                dataset["rotation_axis_trans"][i][3:] = current_xpos

                dataset["quat"][i][:, :4] = rel_quaternion
                dataset["quat"][i][:, 4:] = rel_trans

                dataset["quat_1"][i][:, :4] = rel_quaternion1
                dataset["quat_1"][i][:, 4:] = rel_trans1

                # Collect Log Quaternion data
                dataset["log_quat"][i][:, :4] = calculate_log_quat(rel_quaternion)
                dataset["log_quat"][i][:, 4:] = rel_trans
                dataset["log_quat_1"][i][:, :4] = calculate_log_quat(rel_quaternion1)
                dataset["log_quat_1"][i][:, 4:] = rel_trans1

                # Collect Dual-Quaternion data
                dualQuaternion = get_dualQ(rel_quaternion, rel_trans)
                dualQuat_1 = get_dualQ(rel_quaternion1, rel_trans1)
                dataset["dual_quat"][i] = dualQuaternion
                dataset["dual_quat_1"][i] = dualQuat_1

                # Collect log_dualQ data (= bivector = rotation axis)
                dataset["log_dualQ"][i] = logDual(dualQuaternion)
                dataset["log_dualQ_1"][i] = logDual(dualQuat_1)

                dataset["pos_diff_start"][i] = current_pos - start_xyz
                dataset["pos_diff_1"][i] = current_pos - prev_pos
                prev_pos = current_pos

            # # Relative to origin centered cube.
            # dataset["rot_mat_ori"][i][:, :9] = current_rotMat.flatten()
            # dataset["rot_mat_ori"][i][:, 9:] = current_xpos

            # quat = get_quat(data, body_id)
            # dataset["quat_ori"][i][:, :4] = quat
            # dataset["quat_ori"][i][:, 4:] = current_xpos

            # dataset["log_quat_ori"][i][:, :4] = calculate_log_quat(quat)
            # dataset["log_quat_ori"][i][:, 4:] = current_xpos

            # dual_quat = get_dualQ(quat, current_xpos)
            # dataset["dual_quat_ori"][i] = dual_quat
            # dataset["log_dualQ_ori"][i] = logDual(dual_quat)

        else:
            print("Visualisation failed")
            break

    if visualize:
        viewer.close()

    # Check for any NaNs in the generated data
    for key, data_part in dataset.items():
        assert not np.any(
            np.isnan(data_part)
        ), f"Encountered NaN in {key}. Try to recreate dataset {dict_name}.\n Number of NaNs: {np.sum(np.isnan(data_part))}"

    return dataset


def get_sizes(symmetry, bigBlocks):
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
        sizes = [
            np.random.uniform(0.05, 0.5),
            np.random.uniform(0.05, 0.5),
            np.random.uniform(0.05, 0.5),
        ]
        if bigBlocks:
            sizes *= 10
        return f"{sizes[0]} {sizes[1]} {sizes[2]}", sizes
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
    random_size = np.random.uniform(0.05, 0.5)
    if bigBlocks:
        random_size *= 10
    sizes = ratio * random_size
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
    os.makedirs("data", exist_ok=True)
    dir = f"data/data_t{str(vel_range_l).replace(' ', '')}_r{str(vel_range_a).replace(' ', '')}_{symmetry}_p{plane}_g{grav}"
    if tennis_effect:
        dir = f"data/data_{symmetry}_p{plane}_g{grav}_tennisEffect"

    os.makedirs(dir, exist_ok=True)
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
        plane_str = (
            '<geom type="plane" pos="0 0 0" size="1000 1000 1000" rgba="1 1 1 1"/>'
        )
    else:
        plane_str = ""

    if gravity:
        gravity_str = f'<option integrator="{integrator}"/>'
    else:
        gravity_str = f'<option integrator="{integrator}" gravity="0 0 0"/>'
    return f"""
    <mujoco>
    {gravity_str}
    <worldbody>
        <light name="top" pos="0 0 1"/>
        <camera name="camera2" pos="10 -70 70" xyaxes="1 0 0 0 1 1.5"/>
        <body name="object_body" euler="{euler_obj}" pos="{pos_obj}">
            <joint name="joint1" type="free"/>
            <geom name="object_geom" type="box" size="{size_obj}" rgba="0.7 0.7 0.3 1"/>
        </body>
        <camera name="trackingCamera" pos="0 0 1" fovy="45" mode="targetbody" target="object_body" />
        {plane_str}
    </worldbody>
    </mujoco>
    """


def get_pos(symmetry, gravity, plane, sizes_list):
    if (gravity and plane) or plane:
        if symmetry == "full":
            min_z_position = sizes_list[0]
        elif symmetry == "semi":
            min_z_position = 6 * sizes_list[0]
        elif symmetry == "tennis":
            min_z_position = 5.7 * sizes_list[0]
        else:
            min_z_position = max(sizes_list)  # TODO
        pos = f"{np.random.uniform(-10, 10)} {np.random.uniform(-10, 10)} {np.random.uniform(min_z_position, 2 *min_z_position)}"
    else:
        pos = f"{np.random.uniform(-10, 10)} {np.random.uniform(-10, 10)} {np.random.uniform(-10, 10)}"
    return pos


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
    bigBlocks,
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
    # Warn if directory already exists with more simulations.
    if len(os.listdir(dir)) > num_sims:
        print(
            f"This directory already existed with {len(os.listdir(dir))} files, you want {num_sims} files. Please delete directory {dir}."
        )
        raise IndexError(
            f"This directory ({dir}) already exists with fewer simulations."
        )
    if tennis_effect and not symmetry == "tennis":
        raise BaseException("Cannot create tenniseffect if symmetry is not tennis.")
    for sim_id in range(num_sims):
        if sim_id % 100 == 0 or sim_id == num_sims - 1:
            print(f"Generating sim {sim_id}/{num_sims-1}")
        # Define euler angle
        euler = f"{np.random.uniform(0, 360)} {np.random.uniform(0, 360)} {np.random.uniform(0, 360)}"
        # Define sizes
        sizes_str, sizes_list = get_sizes(symmetry, bigBlocks)
        # Define position
        pos = get_pos(symmetry, gravity, plane, sizes_list)
        string = get_string(euler, pos, sizes_str, gravity, plane, integrator)
        # Create dataset
        plane_grav = plane and gravity
        dataset = generate_data(
            string,
            n_steps,
            dir,
            plane_grav,
            visualize,
            vel_range_l,
            vel_range_a,
            tennis_effect,
        )
        sim_data = {
            "vars": {
                "euler": euler,
                "pos": pos,
                "sizes": sizes_list,
                "gravity": gravity,
                "n_steps": n_steps,
                "symm": symmetry,
                "tennis_effect": tennis_effect,
            },
            "data": dataset,
        }
        # Write data to file
        with open(f"{dir}/sim_{sim_id}.pickle", "wb") as f:
            pickle.dump(sim_data, f)


if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("-n_sims", type=int, help="number of simulations", default=2400)
    parser.add_argument("-n_frames", type=int, help="number of frames", default=500)
    parser.add_argument(
        "-s",
        "--symmetry",
        type=str,
        choices=["full", "semi", "tennis", "none"],
        help="symmetry of the box.\nfull: symmetric box 1:1:1\n; semi: 2 sides of same length, other longer 1:1:10\n;tennis: tennis_racket effect 1:3:10\n;none: random lengths for each side",
        default="tennis",
    )
    parser.add_argument("-l_min", type=int, help="linear qvel min", default=5)
    parser.add_argument("-l_max", type=int, help="linear qvel max", default=20)
    parser.add_argument("-a_min", type=int, help="angular qvel min", default=5)
    parser.add_argument("-a_max", type=int, help="angular qvel max", default=20)
    parser.add_argument(
        "-i",
        "--integrator",
        type=str,
        choices=["RK4", "Euler"],
        help="type of integrator to use",
        default="Euler",
    )
    parser.add_argument("--data_dir", type=str, help="Name of data directory")
    parser.add_argument("--gravity", action=argparse.BooleanOptionalAction)
    parser.add_argument("--plane", action=argparse.BooleanOptionalAction)
    parser.add_argument("--visualize", action=argparse.BooleanOptionalAction)
    parser.add_argument("--tennis_effect", action=argparse.BooleanOptionalAction)
    parser.add_argument("--big", action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    vel_range_l = (args.l_min, args.l_max)
    vel_range_a = (args.a_min, args.a_max)

    print(
        f"Creating dataset vel_range_l={vel_range_l}, vel_range_a={vel_range_a}, symmetry={args.symmetry}"
    )
    print(args.data_dir)
    if not args.data_dir:
        data_dir = get_dir(
            vel_range_l,
            vel_range_a,
            args.symmetry,
            args.n_sims,
            args.plane,
            args.gravity,
            args.tennis_effect,
        )
    else:
        print("here")
        os.makedirs("data", exist_ok=True)
        os.makedirs(f"data/{args.data_dir}", exist_ok=True)
        data_dir = "data/" + args.data_dir

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
        bigBlocks=args.big,
    )
    print(f"Saved in {data_dir}")
    print(f"\nTime: {time.time()- start_time}\n---- FINISHED ----")
