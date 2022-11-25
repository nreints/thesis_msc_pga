from platform import architecture
import torch
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from torch_nn import fcnn
from lstm import LSTM
import pickle
from random import randint
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from convert import *
import math

# from pyquaternion import Quaternion

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
    print("Current model: \n", model)

    return model, config

def get_random_sim_data(data_type, nr_frames):
    # Select random simulation
    i = randint(0, 749)

    print("Using simulation number ", i)
    with open(f'data/sim_{i}.pickle', 'rb') as f:
        file = pickle.load(f)
        if data_type == "pos_diff_start":
            start_pos = torch.tensor(file["data"]["pos"][0], dtype=torch.float32).flatten()
            start_pos = start_pos[None, :].repeat(nr_frames, 1, 1)
        else:
            start_pos = torch.tensor(file["data"]["start"], dtype=torch.float32).flatten()
            start_pos = start_pos[None, :].repeat(nr_frames, 1, 1)

        original_data = torch.tensor(file["data"][data_type], dtype=torch.float32).flatten(start_dim=1)

        # Convert to pos data for plotting
        plot_data = convert(original_data, start_pos, data_type).reshape(nr_frames, 8, 3)
        plot_data2 = torch.tensor(file["data"]["pos"], dtype=torch.float32).reshape(nr_frames, 8, 3)

    print(plot_data.shape)
    return plot_data, original_data, plot_data2, start_pos[0]

def get_prediction_fcnn(original_data, data_type, xyz_data, start, nr_input_frames):
    result = torch.zeros_like(xyz_data)

    # Get first position
    start_pos = start[None, :]

    for frame_id in range(20, xyz_data.shape[0]):
        # Get 20 frames shape: (1, 480)
        input_data = original_data[frame_id - nr_input_frames : frame_id]

        input_data = input_data.unsqueeze(dim=0)
        input_data = input_data.flatten(start_dim=1)

        # Save the prediction in result
        with torch.no_grad(): # Deactivate gradients for the following code
            prediction = model(input_data)
            # print(input_data)
            # # print()
            # print(prediction)
            # exit()
            result[frame_id] = convert(prediction, start_pos, data_type).reshape(-1, 8, 3)

    return result

def get_prediction_lstm(original_data, data_type, xyz_data, start, nr_frames, out_is_in=False):
    # Collect prediction of model given simulation
    # Result should be xyz data for plot
    frames, vert, dim = xyz_data.shape

    # Because LSTM predicts 1 more frame
    result = torch.zeros((frames+1, vert, dim))


    # Get first position
    start_pos = start[None, :]
    hidden = torch.zeros(1, 1, 96)
    cell = torch.zeros(1, 1, 96) #TODO

    for frame_id in range(0, xyz_data.shape[0], nr_frames):
        # Get 20 frames shape: (1, 480)
        if not out_is_in or frame_id == 0:
            input_data = original_data[frame_id : frame_id + nr_frames]
            input_data = input_data.unsqueeze(dim=0)

        # print("in", input_data.shape)
        # Save the prediction in result
        with torch.no_grad(): # Deactivate gradients for the following code
            prediction, (hidden, cell) = model(input_data, (hidden, cell))
            # print("pred", prediction.shape)
            if out_is_in:
                input_data = prediction

            out_shape = result[frame_id + 1 : frame_id + 21].shape
            # print("out", convert(prediction, start_pos, data_type).reshape(-1, 8, 3)[:out_shape[0], :, :].shape)
            result[frame_id + 1 : frame_id + nr_frames + 1] = convert(prediction, start_pos, data_type).reshape(-1, 8, 3)[:out_shape[0], :, :]

    return result


def calculate_edges(cube):
    list_ind = [1, 3, 2, 0, 4, 6, 2, 3, 7, 5, 1, 5, 4, 6, 7]
    edges = [cube[i, :] for i in list_ind]
    return np.append(cube[0, :], edges).reshape(-1,3)

def distance_check(converted, predicted, check):
    X_conv, Y_conv, Z_conv = converted[:, 0], converted[:, 1], converted[:, 2]
    X_pred, Y_pred, Z_pred = predicted[:, 0], predicted[:, 1], predicted[:, 2]
    X_check, Y_check, Z_check = check[:, 0], check[:, 1], check[:, 2]

    distance_conv = ((X_conv[0] - X_conv[1])**2 + (Y_conv[0] - Y_conv[1])**2 + (Z_conv[0] - Z_conv[1])**2)**0.5
    distance_predicted = ((X_pred[0] - X_pred[1])**2 + (Y_pred[0] - Y_pred[1])**2 + (Z_pred[0] - Z_pred[1])**2)**0.5
    distance_check = ((X_check[0] - X_check[1])**2 + (Y_check[0] - Y_check[1])**2 + (Z_check[0] - Z_check[1])**2)**0.5

    print(distance_check, distance_predicted)
    assert math.isclose(distance_conv, distance_predicted, abs_tol=0.001)

def plot_3D_animation(data, result, real_pos_data, data_type, architecture):
    print("data", data.shape, result.shape, real_pos_data.shape)
    data = data
    result = result

    # Open figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Collect init data
    converted_cube = data[0]
    predicted_cube = result[0]
    check_cube = real_pos_data[0]

    #  TODO
    # distance_check(converted_cube, predicted_cube, check_cube)

    # Scatter the corners
    ax.scatter(converted_cube[:, 0], converted_cube[:, 1], converted_cube[:, 2], linewidth=0.5, color='b', label="converted pos")
    ax.scatter(predicted_cube[:, 0], predicted_cube[:, 1], predicted_cube[:, 2], color='r', label="prediction")
    ax.scatter(check_cube[:, 0], check_cube[:, 1], check_cube[:, 2], color="black", label="real pos")

    # Calculate the edges
    converted_cube_edges = calculate_edges(converted_cube)
    predicted_cube_edges = calculate_edges(predicted_cube)
    check_cube_edges = calculate_edges(check_cube)

    # Plot the edges
    ax.plot(converted_cube_edges[:, 0], converted_cube_edges[:, 1], converted_cube_edges[:, 2], c="b")
    # TODO error?
    # ax.plot(predicted_cube_edges[:, 0], predicted_cube_edges[:, 1], predicted_cube_edges[:, 2], c="r")
    # ax.plot(check_cube_edges[:][0], check_cube_edges[:][1], check_cube_edges[:][2], c="black")

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

        # TODO
        # distance_check(converted_cube, predicted_cube, check_cube)

        # Scatter vertice data
        ax.scatter(converted_cube[:, 0], converted_cube[:, 1], converted_cube[:, 2], color='b', linewidth=0.5)
        ax.scatter(predicted_cube[:, 0], predicted_cube[:, 1], predicted_cube[:, 2], color='r', linewidth=0.5)
        ax.scatter(check_cube[:, 0], check_cube[:, 1], check_cube[:, 2], color='black', linewidth=0.5)

        # Calculate the edges
        converted_cube_edges = calculate_edges(converted_cube)
        predicted_cube_edges = calculate_edges(predicted_cube)
        check_cube_edges = calculate_edges(check_cube)

        # Plot the edges
        ax.plot(converted_cube_edges[:, 0], converted_cube_edges[:, 1], converted_cube_edges[:, 2], label="converted pos")
        ax.plot(predicted_cube_edges[:, 0], predicted_cube_edges[:, 1], predicted_cube_edges[:, 2], c="r", label="prediction")
        ax.plot(check_cube_edges[:, 0], check_cube_edges[:, 1], check_cube_edges[:, 2], c="black", label="real pos")

        ax.set_xlim3d(-15, 15)
        ax.set_ylim3d(-15, 15)
        ax.set_zlim3d(0, 40)

        ax.set_xlabel('$X$')
        ax.set_ylabel('$Y$')
        ax.set_zlabel('$Z$')
        ax.set_title(f"{data_type} trained with {architecture}")
        ax.legend()

    # Interval : Delay between frames in milliseconds.
    ani = animation.FuncAnimation(fig, update, 225, interval=100, repeat=False)

    plt.show()


if __name__ == "__main__":
    nr_frames = 225 # See new_mujoco.py
    data_type = "quat"
    architecture = "lstm"

    model, config = load_model(data_type, architecture)

    nr_input_frames = config["n_frames"]
    plot_data, ori_data, pos_data, start = get_random_sim_data(data_type, nr_frames)

    if architecture == "fcnn":
        prediction = get_prediction_fcnn(ori_data, data_type, plot_data, start, nr_input_frames)
    elif architecture == "lstm":
        prediction = get_prediction_lstm(ori_data, data_type, plot_data, start, 5, out_is_in=False)

    plot_3D_animation(np.array(plot_data), np.array(prediction), np.array(pos_data), data_type, architecture)
