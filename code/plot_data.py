from platform import architecture
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
import math


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def load_model(data_type, architecture):
    # Load model
    model_dict = torch.load(f"models/{data_type}_{architecture}.pickle", map_location=torch.device(device))
    config = model_dict['config']
    ndata_dict = model_dict['data_dict']

    if architecture == "fcnn":
        model = fcnn(ndata_dict[config['data_type']], config)
    elif architecture == "lstm":
        model = LSTM(ndata_dict[config['data_type']], config)

    model.load_state_dict(model_dict['model'])
    model.eval()

    return model, config

def get_random_sim_data(data_type, nr_frames, nr_sims, i=None):
    """
    Collects the data from a random simulation.
    Input:
        - data_type: type of the data that needs to be collected.
        - nr_frames: number of frames to collect.
        - nr_sims: total number of available simulations ().
        - i: id of simulation to select, default; select random simulation.
    Output:
        - plot_data; xyz data converted from data in data_type.
        - original_data; data in the format of data_type.
        - plot_data_true_pos; original xyz data.
        - start_pos[0]; start position (xyz) of the simulation.
    """
    # Select random simulation
    if not i:
        i = randint(0, nr_sims)
    else:
        raise Exception("No simulation id selected. Please enter a valid simulation id")

    print("Using simulation number ", i)
    with open(f'data/sim_{i}.pickle', 'rb') as f:
        file = pickle.load(f)
        # Load the correct start position repeat for converting
        if data_type == "pos_diff_start":
            start_pos = torch.tensor(file["data"]["pos"][0], dtype=torch.float32).flatten()
            start_pos = start_pos[None, :].repeat(nr_frames, 1, 1)
        else:
            start_pos = torch.tensor(file["data"]["start"], dtype=torch.float32).flatten()
            start_pos = start_pos[None, :].repeat(nr_frames, 1, 1)

        # Load the data in correct data type
        original_data = torch.tensor(file["data"][data_type], dtype=torch.float32).flatten(start_dim=1)

        # Convert to xyz position data for plotting
        plot_data = convert(original_data, start_pos, data_type).reshape(nr_frames, 8, 3)
        # Load original xyz position data for validating plot_data
        plot_data_true_pos = torch.tensor(file["data"]["pos"], dtype=torch.float32).reshape(nr_frames, 8, 3)

    return plot_data, original_data, plot_data_true_pos, start_pos[0]

def get_prediction_fcnn(original_data, data_type, xyz_data, start_pos, nr_input_frames, model):
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
        - result: converted to xyz positions output of the model based on original_data and start_pos.
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
            result[frame_id] = convert(prediction, start_pos, data_type).reshape(-1, 8, 3)

    return result

def get_prediction_lstm(original_data, data_type, xyz_data, start_pos, nr_input_frames, model, out_is_in=False):
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
        - result: converted to xyz positions output of the model based on original_data and start_pos.
    """
    # Result should be xyz data for plot
    frames, vert, dim = xyz_data.shape

    # Because LSTM predicts 1 more frame
    result = torch.zeros((frames + 1, vert, dim))

    # Get first position
    start_pos = start_pos[None, :]
    hidden = torch.zeros(1, 1, 96)
    cell = torch.zeros(1, 1, 96) #TODO

    for frame_id in range(0, xyz_data.shape[0], nr_input_frames):
        # Get 20 frames shape: (1, 480)
        if not out_is_in or frame_id == 0:
            input_data = original_data[frame_id : frame_id + nr_input_frames]
            input_data = input_data.unsqueeze(dim=0)

        # Save the prediction in result
        with torch.no_grad(): # Deactivate gradients for the following code
            prediction, (hidden, cell) = model(input_data, (hidden, cell))
            if out_is_in:
                input_data = prediction

            out_shape = result[frame_id + 1 : frame_id + nr_input_frames + 1].shape
            result[frame_id + 1 : frame_id + nr_input_frames + 1] = convert(prediction, start_pos, data_type).reshape(-1, 8, 3)[:out_shape[0], :, :]

    return result

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
    edges = np.append(cube[0, :], edges_part).reshape(-1,3)
    return edges

def distance_check(converted, check):
    """
    Checks whether the converted cube is close to the validation cube.
    Input:
        - converted: the xyz vertice positions of the converted cube.
        - check: the xyz vertice positions of the validation cube.
    Output: None
    """
    X_conv, Y_conv, Z_conv = converted[:, 0], converted[:, 1], converted[:, 2]
    # X_pred, Y_pred, Z_pred = predicted[:, 0], predicted[:, 1], predicted[:, 2]
    X_check, Y_check, Z_check = check[:, 0], check[:, 1], check[:, 2]

    distance_conv = ((X_conv[0] - X_conv[1])**2 + (Y_conv[0] - Y_conv[1])**2 + (Z_conv[0] - Z_conv[1])**2)
    # distance_predicted = ((X_pred[0] - X_pred[1])**2 + (Y_pred[0] - Y_pred[1])**2 + (Z_pred[0] - Z_pred[1])**2)**0.5
    distance_check = ((X_check[0] - X_check[1])**2 + (Y_check[0] - Y_check[1])**2 + (Z_check[0] - Z_check[1])**2)

    assert math.isclose(distance_conv, distance_check, rel_tol=0.0001)

def plot_cubes(conv_cube, pred_cube, check_cube, ax):
    """
    Plots the cubes.
    """
    # Scatter the corners
    ax.scatter(conv_cube[:, 0], conv_cube[:, 1], conv_cube[:, 2], linewidth=0.5, color='b', label="converted pos")
    ax.scatter(pred_cube[:, 0], pred_cube[:, 1], pred_cube[:, 2], color='r', label="prediction")
    ax.scatter(check_cube[:, 0], check_cube[:, 1],  check_cube[:, 2], color="black", label="real pos")

    # Calculate the edges
    converted_cube_edges = calculate_edges(conv_cube)
    predicted_cube_edges = calculate_edges(pred_cube)
    check_cube_edges = calculate_edges(check_cube)

    # Plot the edges
    ax.plot(converted_cube_edges[:, 0], converted_cube_edges[:, 1], converted_cube_edges[:, 2], c="b")
    ax.plot(predicted_cube_edges[:, 0], predicted_cube_edges[:, 1], predicted_cube_edges[:, 2], c="r")
    ax.plot(check_cube_edges[:, 0], check_cube_edges[:, 1], check_cube_edges[:, 2], c="black")


def plot_3D_animation(data, result, real_pos_data, data_type, architecture, nr_frames):
    """
    Plots 3D animation of the cubes.
    Input:
        - data: converted xyz vertice positions.
        - result: predicted xyz vertice positions.
        - real_pos_data: original xyz vertice positions.
        - data_type: original data type of data.
        - architecture: architecture of the pretrained model.
        - nr_frames: total number of frames in the simulation.
    """
    # Open figure
    fig = plt.figure()
    fig.suptitle(f"{data_type} trained with {architecture}")
    ax = fig.add_subplot(111, projection='3d')

    # Collect init data
    converted_cube = data[0]
    predicted_cube = result[0]
    check_cube = real_pos_data[0]

    distance_check(converted_cube, check_cube)

    plot_cubes(converted_cube, predicted_cube, check_cube, ax)

    ax.set_xlim3d(-15, 15)
    ax.set_ylim(-15, 15)
    ax.set_zlim(0, 50)

    def update(idx):
        # Remove the previous scatter plot
        if idx != 0:
            ax.cla()

        # Get cube vertice data
        converted_cube = data[idx]
        predicted_cube = result[idx]
        check_cube = real_pos_data[idx]

        distance_check(converted_cube, check_cube)

        plot_cubes(converted_cube, predicted_cube, check_cube, ax)

        ax.set_xlim3d(-15, 15)
        ax.set_ylim3d(-15, 15)
        ax.set_zlim3d(0, 40)

        ax.set_xlabel('$X$')
        ax.set_ylabel('$Y$')
        ax.set_zlabel('$Z$')
        ax.set_title(f"Frame {idx}/{nr_frames}")
        ax.legend()

    # Interval : Delay between frames in milliseconds.
    ani = animation.FuncAnimation(fig, update, nr_frames, interval=75, repeat=False)
    plt.show()


def plot_datatypes(plot_data, data_types):
    """
    Plots 3D animation of the cubes in all data types
    """
    # Open figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = ["b", "g", "r", "m", "k", "c"]
    for i in range(len(data_types)):
        # Collect init data
        converted_cube = np.array(plot_data[i][0])

        # Scatter the corners
        ax.scatter(converted_cube[:, 0], converted_cube[:, 1], converted_cube[:, 2], linewidth=0.5, color=colors[i], label=data_types[i])

        # Calculate the edges
        converted_cube_edges = calculate_edges(converted_cube)

        # Plot the edges
        ax.plot(converted_cube_edges[:, 0], converted_cube_edges[:, 1], converted_cube_edges[:, 2], c=colors[i])


    ax.set_xlim3d(-15, 15)
    ax.set_ylim(-15, 15)
    ax.set_zlim(0, 50)
    ax.legend()

    def update(idx):

        # Remove the previous scatter plot
        if idx != 0:
            ax.cla()

        if idx % 10 == 0:
            print("step ", idx)

        for i in range(len(data_types)):

            # Get cube vertice data
            converted_cube = np.array(plot_data[i][idx])

            # Scatter vertice data
            ax.scatter(converted_cube[:, 0], converted_cube[:, 1], converted_cube[:, 2], color=colors[i], linewidth=0.5)

            # Calculate the edges
            converted_cube_edges = calculate_edges(converted_cube)

            # Plot the edges
            ax.plot(converted_cube_edges[:, 0], converted_cube_edges[:, 1], converted_cube_edges[:, 2], label=data_types[i], color=colors[i])

        ax.set_xlim3d(-15, 15)
        ax.set_ylim3d(-15, 15)
        ax.set_zlim3d(0, 40)

        ax.set_xlabel('$X$')
        ax.set_ylabel('$Y$')
        ax.set_zlabel('$Z$')
        ax.set_title(f"All Datatypes converted to xyz-position")
        ax.legend()

    # Interval : Delay between frames in milliseconds.
    ani = animation.FuncAnimation(fig, update, nr_frames, interval=50, repeat=False)

    plt.show()
    plt.close()


if __name__ == "__main__":
    nr_frames = 250 # See create_data.py
    nr_sims = 5000
    # ["pos", "eucl_motion", "quat", "log_quat", "dual_quat", "log_dualQ", "pos_diff_start"]
    data_type = "log_dualQ"
    architecture = "lstm"
    print(f"Visualizing {architecture} trained on {data_type}")

    model, config = load_model(data_type, architecture)
    plot_data, ori_data, pos_data, start = get_random_sim_data(data_type, nr_frames, nr_sims)

    nr_input_frames = config["n_frames"]
    if architecture == "fcnn":
        prediction = get_prediction_fcnn(ori_data, data_type, plot_data, start, nr_input_frames, model)
    elif architecture == "lstm":
        prediction = get_prediction_lstm(ori_data, data_type, plot_data, start, nr_input_frames, model, out_is_in=False)

    plot_3D_animation(np.array(plot_data), np.array(prediction), np.array(pos_data), data_type, architecture, nr_frames)



    # # Below the test for all datatypes
    # plot_data = []
    # pos_data = []
    # i = randint(0, nr_sims-1)

    # # Test all data types:
    # data_types = ["pos", "eucl_motion", "quat", "log_quat", "dual_quat", "pos_diff_start"]
    # for data_thing in data_types:
    #     result = get_random_sim_data(data_thing, nr_frames, nr_sims, i)
    #     plot_data.append(result[0])

    # plot_datatypes(plot_data, data_types)


