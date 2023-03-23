import torch
import matplotlib.pyplot as plt
from fcnn import fcnn
from lstm import LSTM
import pickle
from random import randint
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from convert import *
import argparse
import os
import matplotlib.cm as cm

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def load_model(data_type, architecture, data_dir):
    """
    Loads a pretrained model.

    Input:
        - data_type: current data type.
        - architecture: architecture of the pretrained model.

    Output:
        - model: pretrained model.
        - config: config of the model.
    """
    # Load model
    model_dict = torch.load(
        f"models/{architecture}/{data_type}_{architecture}_['{data_dir}'].pickle",
        map_location=torch.device(device),
    )
    config = model_dict["config"]
    ndata_dict = model_dict["data_dict"]

    if architecture == "fcnn":
        model = fcnn(ndata_dict[config["data_type"]], config)
    elif architecture == "lstm" or architecture == "quaternet":
        model = LSTM(ndata_dict[config["data_type"]], config)

    model.load_state_dict(model_dict["model"])
    model.eval()

    return model, config


def get_random_sim_data(data_type, nr_sims, data_dir, i=None):
    """
    Collects the data from a random simulation.

    Input:
        - data_type: type of the data that needs to be collected.
        - nr_sims: total number of available simulations ().
        - i: id of simulation to select, default; select random simulation.

    Output:
        - plot_data; xyz data converted from data in data_type.
        - original_data; data in the format of data_type.
        - plot_data_true_pos; original xyz data.
        - start_pos[0]; start position (xyz) of the simulation.
        - nr_frames: number of frames to collect.
    """
    # Select random simulation
    if i is None:
        i = randint(0, nr_sims - 1)
        print(f"Using random simulation number {i}, data_type {data_type}")
    else:
        print(f"Using simulation {i}, data_type {data_type}")

    with open(f"{data_dir}/sim_{i}.pickle", "rb") as f:
        file = pickle.load(f)
        nr_frames = file["vars"]["n_steps"]
        # Load the correct start position repeat for converting
        if data_type[-3:] == "ori":
            print("using old way")
            start_pos = torch.tensor(
                file["data"]["start"], dtype=torch.float32
            ).flatten()
            start_pos = start_pos[None, :].repeat(nr_frames, 1, 1)
        else:
            start_pos = torch.tensor(
                file["data"]["pos"][0], dtype=torch.float32
            ).flatten()
            start_xpos = torch.tensor(
                file["data"]["xpos_start"], dtype=torch.float32
            ).flatten()
            start_pos = start_pos[None, :].repeat(nr_frames, 1, 1)
            start_xpos = start_xpos[None, :].repeat(nr_frames, 1, 1)

        if "rotation_axis_trans" in file["data"].keys():
            rot_axis_trans = file["data"]["rotation_axis_trans"]
        else:
            print("NO ROTATION AXIS")
            rot_axis_trans = None

        # Load the data in correct data type
        original_data = torch.FloatTensor(file["data"][data_type]).flatten(start_dim=1)
        if data_type == "quat":
            print("shape original_data: ", original_data.shape)
            data_in_file = file["data"][data_type].squeeze()
            print("shape data_in_file: ", data_in_file.shape)
            print(np.any(data_in_file[:, 4:] != 0))
            print(torch.any(original_data[:][4:] != 0))
            exit()
        # if data_type[-3:] == "ori":
        #     data_type = data_type[:-4]
        # Convert to xyz position data for plotting
        if (
            data_type[-3:] != "ori"
            and data_type != "pos"
            and data_type != "pos_diff_start"
        ):
            plot_data = convert(
                original_data, start_pos, data_type, start_xpos
            ).reshape(nr_frames, 8, 3)
        else:
            plot_data = convert(original_data, start_pos, data_type).reshape(
                nr_frames, 8, 3
            )

        # Load original xyz position data for validating plot_data
        plot_data_true_pos = torch.tensor(
            file["data"]["pos"], dtype=torch.float32
        ).reshape(nr_frames, 8, 3)

        ranges = [
            (torch.min(plot_data_true_pos), torch.max(plot_data_true_pos))
            for _ in range(3)
        ]

    return (
        plot_data,
        original_data,
        plot_data_true_pos,
        start_pos[0],
        nr_frames,
        i,
        rot_axis_trans,
        ranges,
    )


def get_prediction_fcnn(
    original_data, data_type, xyz_data, start_pos, nr_input_frames, model
):
    """
    Gets prediction of the pre-trained fcnn.

    Input:
        - original_data: input data in data_type.
        - data_type: data type currently used.
        - xyz_data: xyz data.
        - start_pos: start position of the simulation.
        - nr_input_frames: number of frames the fcnn is trained on.
        - model: the trained model.

    Output:
        - prediction: converted to xyz positions output of the model based on original_data and start_pos.
    """
    result = torch.zeros_like(xyz_data)

    for frame_id in range(nr_input_frames, xyz_data.shape[0]):
        # Get nr_input_frames frames shape: (nr_input_frames, n_data)
        input_data = original_data[frame_id - nr_input_frames : frame_id]
        # Reshape to (1, nr_input_frames*n_data)
        input_data = input_data.unsqueeze(dim=0).flatten(start_dim=1)

        # Save the prediction in result
        with torch.no_grad():
            prediction = model(input_data)
            result[frame_id] = convert(prediction, start_pos, data_type).reshape(
                -1, 8, 3
            )

    return result


def get_prediction_lstm(
    original_data,
    data_type,
    xyz_data,
    start_pos,
    nr_input_frames,
    model,
    out_is_in=False,
):
    """
    Gets prediction of the pre-trained lstm.

    Input:
        - original_data: input data in data_type.
        - data_type: data type currently used.
        - xyz_data: xyz data.
        - start_pos: start position of the simulation.
        - nr_input_frames: number of frames the fcnn is trained on.
        - model: the trained model.
        - out_is_in:
                    False; do not use output of the model as input.
                    True; do use output of the model as input.

    Output:
        - prediction: converted to xyz positions output of the model based on original_data and start_pos.
    """
    # prediction should be xyz data for plot
    frames, vert, dim = xyz_data.shape

    # Because LSTM predicts 1 more frame
    result = torch.zeros((frames + 1, vert, dim))

    # Get first position
    start_pos = start_pos[None, :]
    hidden = torch.zeros(1, 1, 96)
    cell = torch.zeros(1, 1, 96)  # TODO

    for frame_id in range(0, xyz_data.shape[0], nr_input_frames):
        # Get 20 frames shape: (1, 480)
        if not out_is_in or frame_id == 0:
            input_data = original_data[frame_id : frame_id + nr_input_frames]
            input_data = input_data.unsqueeze(dim=0)

        # Save the prediction in result
        with torch.no_grad():  # Deactivate gradients for the following code
            prediction, (hidden, cell) = model(input_data, (hidden, cell))
            if out_is_in:
                input_data = prediction

            out_shape = prediction[frame_id + 1 : frame_id + nr_input_frames + 1].shape
            result[frame_id + 1 : frame_id + nr_input_frames + 1] = convert(
                prediction, start_pos, data_type
            ).reshape(-1, 8, 3)[: out_shape[0], :, :]

    return result


def distance_check(converted, check):
    """
    Checks whether the converted cube is close to the validation cube.

    Input:
        - converted: the xyz vertice positions of the converted cube.
        - check: the xyz vertice positions of the validation cube.

    Output: assertion
    """
    assert np.allclose(converted, check, atol=1e-4)


def calculate_edges(cube):
    """
    Determines the edges of a cube.

    Input:
        - cube: xyz position of the vertices of the cube.

    Output:
        - edges: the edges of the cube.
    """
    list_ind = [1, 3, 2, 0, 4, 6, 2, 3, 7, 5, 1, 5, 4, 6, 7]
    edges_part = [cube[i, :] for i in list_ind]
    edges = np.append(cube[0, :], edges_part).reshape(-1, 3)
    return edges


def plot_cubes(conv_cube, pred_cube, check_cube, ax):
    """
    Plots the cubes.

    Input:
        - conv_cube; xyz-position of vertices converted from original data_type.
        - pred_cube; predicted xyz-position of vertices by network.
        - check_cube; real xyz-position of vertices.

    Output:
        - plots the cubes
    """
    plot_cube(conv_cube, ax, "converted", "b")
    plot_cube(pred_cube, ax, "predicted", "r")
    plot_cube(check_cube, ax, "real pos", "black")


def plot_cube(cube_data, ax, label, color_cube):
    ax.scatter(
        cube_data[:, 0],
        cube_data[:, 1],
        cube_data[:, 2],
        linewidth=0.5,
        color=color_cube,
        label=label,
    )
    cube_edges = calculate_edges(cube_data)
    ax.plot(cube_edges[:, 0], cube_edges[:, 1], cube_edges[:, 2], c=color_cube)


def plot_3D_animation(
    data,
    prediction,
    real_pos_data,
    data_type,
    architecture,
    nr_frames,
    sim_id,
    data_dir,
    range_plot,
):
    """
    Plots 3D animation of the cubes.

    Input:
        - data: converted xyz vertice positions.
        - prediction: predicted xyz vertice positions.
        - real_pos_data: original xyz vertice positions.
        - data_type: original data type of data.
        - architecture: architecture of the pretrained model.
        - nr_frames: total number of frames in the simulation.
    """
    # Open figure
    fig = plt.figure()
    fig.suptitle(f"{data_type} trained with {architecture}")
    ax = fig.add_subplot(111, projection="3d")

    # Collect init data
    converted_cube = data[0]
    predicted_cube = prediction[0]
    check_cube = real_pos_data[0]

    distance_check(converted_cube, check_cube)

    plot_cubes(converted_cube, predicted_cube, check_cube, ax)

    set_ax_properties(ax, 0, sim_id, data_dir, range_plot)

    def update(idx):
        # Remove the previous scatter plot
        if idx != 0:
            ax.cla()

        # Get cube vertice data
        converted_cube = data[idx]
        predicted_cube = prediction[idx]
        check_cube = real_pos_data[idx]

        distance_check(converted_cube, check_cube)

        plot_cubes(converted_cube, predicted_cube, check_cube, ax)

        set_ax_properties(ax, idx, sim_id, data_dir, range_plot)

    # Interval : Delay between frames in milliseconds.
    ani = animation.FuncAnimation(fig, update, nr_frames, interval=75, repeat=False)
    plt.show()


def plot_datatype_cubes(data_types, plot_data, rot_axis, idx, ax):
    colors = ["b", "g", "r", "m", "k", "c", "b", "b", "g", "r", "m", "k", "c", "b"]
    for i in range(len(data_types)):

        # Get cube vertice data
        converted_cube = np.array(plot_data[i][idx])

        # Scatter vertice data in different colors
        x_values = converted_cube.T[0]
        colors_more = cm.rainbow(np.linspace(0, 1, len(x_values)))
        for s, point in enumerate(converted_cube):
            ax.scatter(point[0], point[1], point[2], color=colors_more[s])

        # Calculate the edges
        converted_cube_edges = calculate_edges(converted_cube)

        # Plot the edges
        ax.plot(
            converted_cube_edges[:, 0],
            converted_cube_edges[:, 1],
            converted_cube_edges[:, 2],
            label=data_types[i],
            color=colors[i],
        )

    rot_axis_current = np.array(rot_axis[i][idx]).reshape(2, 3).T
    # Add translation to rotation vector.
    rot_axis_translated = np.sum(rot_axis_current, axis=1).reshape(
        3,
    )
    # print(rot_axis_current[:, 0], rot_axis_translated)
    # Get direction of rotation axis.
    direction = (
        rot_axis_translated - (np.zeros((3,)) + rot_axis_current[:, -1])
    ).reshape(
        3,
    )
    rot_axis_plot = np.empty_like(rot_axis_current)
    rot_axis_plot[:, 0] = rot_axis_translated - 50 * direction
    rot_axis_plot[:, 1] = rot_axis_translated + 50 * direction

    # ROTATION AXIS
    ax.plot(
        rot_axis_plot[0],
        rot_axis_plot[1],
        rot_axis_plot[2],
        color="g",
        # label="rotation axis",
    )

    # ARROW TO CENTER
    # ax.quiver(0,0,0,rot_axis_current[0, -1],rot_axis_current[1, -1],rot_axis_current[2, -1], color="darkred")

    # DIRECTION ROTATION AXIS
    ax.quiver(
        0,
        0,
        0,
        direction[0] * 10,
        direction[1] * 10,
        direction[2] * 10,
        color="darkviolet",
        # label="direction rotaxis",
    )
    # DIRECTION ROTATION AXIS Translated
    ax.quiver(
        rot_axis_current[0, -1],
        rot_axis_current[1, -1],
        rot_axis_current[2, -1],
        direction[0] * 10,
        direction[1] * 10,
        direction[2] * 10,
        color="darkviolet",
    )

    # ORIGIN
    ax.scatter([0], [0], [0], color="darkred", marker="*")  # , label="origin")

    # X, Y = np.meshgrid(np.arange(-60, 60), np.arange(-60, 60))
    # Z = 0*X
    # ax.plot_surface(Z, Y, X, alpha=0.7)


def set_ax_properties(ax, idx, sim_id, data_dir, range_plot):
    ax.set_xlim3d(range_plot[0][0], range_plot[0][1])
    ax.set_ylim(range_plot[1][0], range_plot[1][1])
    ax.set_zlim(range_plot[2][0], range_plot[2][1])
    ax.legend(bbox_to_anchor=(1.75, 1), ncol=1, fancybox=True)
    ax.set_proj_type("persp", focal_length=0.3)
    ax.set_xlabel("$X$")
    ax.set_ylabel("$Y$")
    ax.set_zlabel("$Z$")
    ax.set_title(f"Frame {idx}/{nr_frames} for sim {sim_id} on set {data_dir[5:]}")


def plot_datatypes(
    plot_data, data_types, nr_frames, rot_axis, sim_id, data_dir, range_plot
):
    """
    Plots 3D animation of the cubes in all data types
    """
    # Open figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    plot_datatype_cubes(data_types, plot_data, rot_axis, 0, ax)

    set_ax_properties(ax, 0, sim_id, data_dir, range_plot)

    def update(idx):

        # Remove the previous scatter plot
        if idx != 0:
            ax.cla()

        plot_datatype_cubes(data_types, plot_data, rot_axis, idx, ax)
        set_ax_properties(ax, idx, sim_id, data_dir, range_plot)

    # Interval : Delay between frames in milliseconds.
    ani = animation.FuncAnimation(
        fig, update, frames=nr_frames, interval=1, repeat=False
    )

    plt.show()
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_type", type=str, help="data type to visualize", default="pos"
    )
    parser.add_argument("-architecture", type=str, help="architecture", default="fcnn")
    parser.add_argument(
        "--data_dir",
        type=str,
        help="data directory",
        default="data_t(0, 0)_r(5, 10)_tennis_pNone_gNone",
    )
    parser.add_argument("--prediction", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    data_dir = "data/" + args.data_dir
    print(f"Using data from directory: {data_dir}")
    if not os.path.exists(data_dir):
        raise KeyError(f"Not such a directory {data_dir}")
    else:
        print("Found the requested data directory.")

    nr_sims = len(os.listdir(data_dir))
    if nr_sims == 0:
        raise KeyError(f"No simulations in {data_dir}")
    else:
        print(f"Found {nr_sims} simulations")

    # -----------------------------------
    if args.prediction:

        data_type = args.data_type
        architecture = args.architecture
        print(f"Visualizing {architecture} trained on {data_type}")

        model, config = load_model(data_type, architecture, args.data_dir)
        (
            plot_data,
            ori_data,
            pos_data,
            start,
            nr_frames,
            sim_id,
            _,
            range_plot,
        ) = get_random_sim_data(data_type, nr_sims, data_dir)

        nr_input_frames = config["n_frames"]
        if architecture == "fcnn":
            prediction = get_prediction_fcnn(
                ori_data, data_type, plot_data, start, nr_input_frames, model
            )
        elif (
            architecture == "lstm"
            or architecture == "quaternet"
            or architecture == "gru"
        ):
            prediction = get_prediction_lstm(
                ori_data,
                data_type,
                plot_data,
                start,
                nr_input_frames,
                model,
                out_is_in=False,
            )

        plot_3D_animation(
            np.array(plot_data),
            np.array(prediction),
            np.array(pos_data),
            data_type,
            architecture,
            nr_frames,
            sim_id,
            args.data_dir,
            range_plot,
        )

    # -----------------------------------
    else:
        # Below the test for all datatypes
        # i = randint(0, nr_sims - 1)
        i = 0
        # print("simulation", i)
        # Test all data types:

        data_types = [
            "pos",
            # "eucl_motion",
            # "eucl_motion_ori",
            "quat",
            # "quat_ori",
            # "log_quat",
            # "log_quat_ori",
            # "dual_quat",
            # "dual_quat_ori",
            # "log_dualQ",
            # "log_dualQ_ori",
            # "pos_diff_start",
        ]
        plot_data, rot_axis, rot_trans_axis = [], [], []

        for data_thing in data_types:
            (
                prediction,
                _,
                _,
                _,
                nr_frames,
                _,
                rotation_axis_trans,
                range_plot,
            ) = get_random_sim_data(data_thing, nr_sims, data_dir, i)
            plot_data.append(prediction)
            rot_trans_axis.append(rotation_axis_trans)

        plot_datatypes(
            plot_data,
            data_types,
            nr_frames,
            rot_trans_axis,
            i,
            args.data_dir,
            range_plot,
        )
