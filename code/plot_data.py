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

# from pyquaternion import Quaternion


def load_model(data_type, architecture):
    # Load model

    model_dict = torch.load(f"models/{data_type}_{architecture}.pickle")
    config = model_dict['config']
    ndata_dict = model_dict['data_dict']

    if architecture == "fcnn":
        model = fcnn(ndata_dict[config['data_type']], config)
    elif architecture == "lstm":
        model = LSTM(ndata_dict[config['data_type']], config)

    model.load_state_dict(model_dict['model'])
    model.eval()
    print("Current model: \n", model)

    return model

def get_random_sim_data(data_type, nr_frames):
    # Select random simulation
    i = randint(0, 749)
    i = 10

    print("Using simulation number ", i)
    with open(f'data/sim_{i}.pickle', 'rb') as f:
        file = pickle.load(f)
        if data_type == "pos_diff_start":
            start_pos = torch.tensor(file["data"]["pos"][0], dtype=torch.float32)
            start_pos = start_pos[None, :].repeat(nr_frames, 1, 1)
        else:
            start_pos = torch.tensor(file["data"]["start"], dtype=torch.float32)
            start_pos = start_pos[None, :].repeat(nr_frames, 1, 1)

        data_tensor = torch.tensor(file["data"][data_type], dtype=torch.float32)

        # Get data as data_type
        original_data = data_tensor.flatten(start_dim=1)

        # Convert to pos data for plotting
        # print(data_tensor.shape)
        plot_data = convert(data_tensor.flatten(start_dim=1), start_pos, data_type).reshape(nr_frames, 8, 3)
        plot_data2 = torch.tensor(file["data"]["pos"], dtype=torch.float32).reshape(nr_frames, 8, 3)
        # print(plot_data2[:10])
        # exit()
        # print("--- PLOT ----")
        # print(start_pos)
        # print(original_data[0])
        # print("1\n", plot_data)
        # print("2\n", plot_data2)

    return plot_data, original_data, plot_data2, start_pos[0]

def get_prediction_fcnn(original_data, data_type, xyz_data, start):
    result = torch.zeros_like(xyz_data)

    # Get first position
    start_pos = start[None, :]

    for frame_id in range(20, xyz_data.shape[0]):
        # Get 20 frames shape: (1, 480)
        input_data = original_data[frame_id - 20 : frame_id]

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
    frames, vert_num, dim = xyz_data.shape

    # Because LSTM predicts 1 more frame
    result = torch.zeros((frames+1, vert_num, dim))


    # Get first position
    start_pos = start[None, :]
    hidden = torch.zeros(1, 1, 96)
    cell = torch.zeros(1, 1, 96) #TODO
    # print(hidden.shape)

    for frame_id in range(0, xyz_data.shape[0], nr_frames):
        # Get 20 frames shape: (1, 480)
        if not out_is_in or frame_id == 0:
            input_data = original_data[frame_id : frame_id + nr_frames]
            input_data = input_data.unsqueeze(dim=0)

        # Save the prediction in result
        with torch.no_grad(): # Deactivate gradients for the following code
            prediction, (hidden, cell) = model(input_data, (hidden, cell))
            if out_is_in:
                input_data = prediction

            out_shape = result[frame_id + 1 : frame_id + 21].shape
            result[frame_id + 1 : frame_id + nr_frames + 1] = convert(prediction, start_pos, data_type).reshape(-1, 8, 3)[:out_shape[0], :, :]

    return result


def plot_3D_animation(data, result, plot_data2):
    data = data
    result = result

    # Open figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel('$X$')
    ax.set_ylabel('$Y$')
    ax.set_xlabel('$Z$')

    # Initial plot
    # Converted data
    first_cube = data[0]
    first_cube = first_cube[np.array([0, 1, 2, 3, 4, 5, 6, 7]), :][np.array([0,1,3,2,6,7,5,4]), :]

    # Predicted data
    cube_result = result[0]
    cube_result = cube_result[np.array([0, 1, 2, 3, 4, 5, 6, 7]), :][np.array([0,1,3,2,6,7,5,4]), :]

    # Original xyz data
    check_cube = plot_data2[0]
    check_cube = check_cube[np.array([0, 1, 2, 3, 4, 5, 6, 7]), :][np.array([0,1,3,2,6,7,5,4]), :]

    X, Y, Z = first_cube[:, 0], first_cube[:, 1], first_cube[:, 2]
    X_pred, Y_pred, Z_pred = cube_result[:, 0], cube_result[:, 1], cube_result[:, 2]
    X_check, Y_check, Z_check = check_cube[:, 0], check_cube[:, 1], check_cube[:, 2]

    distance_check = ((X_check[0] - X_check[1])**2 + (Y_check[0] - Y_check[1])**2 + (Z_check[0] - Z_check[1])**2)**0.5

    distance = ((X_pred[0] - X_pred[1])**2 + (Y_pred[0] - Y_pred[1])**2 + (Z_pred[0] - Z_pred[1])**2)**0.5
    print(distance_check)

    # Begin plotting.
    ax.scatter(X, Y, Z, linewidth=0.5, color='b', label="conv pos == label")
    ax.scatter(X_pred, Y_pred, Z_pred, color='r', linewidth=0.5, label="prediction")
    ax.scatter(X_check, Y_check, Z_check, c="black", label="real pos")

    ax.plot(X, Y, Z)
    ax.plot(X_pred, Y_pred, Z_pred, c="r")
    ax.plot(X_check, Y_check, Z_check, c="black")

    ax.set_xlim3d(-15, 15)
    ax.set_ylim(-15, 15)
    ax.set_zlim(0, 50)

    def update(idx):
        ax.set_xlabel('$X$')
        ax.set_ylabel('$Y$')
        ax.set_xlabel('$Z$')

        # Remove the previous scatter plot
        if idx != 0:
            ax.cla()

        # Get original cube data
        cube = data[idx]
        cube = cube[np.array([0, 1, 2, 3, 4, 5, 6, 7]), :][np.array([0,1,3,2,6,7,5,4]), :]

        # Get predicted cube date
        predicted_cube = result[idx]
        predicted_cube = predicted_cube[np.array([0, 1, 2, 3, 4, 5, 6, 7]), :][np.array([0,1,3,2,6,7,5,4]), :]
        X_pred, Y_pred, Z_pred = predicted_cube[:, 0], predicted_cube[:, 1], predicted_cube[:, 2]

        check_cube = plot_data2[idx]
        check_cube = check_cube[np.array([0, 1, 2, 3, 4, 5, 6, 7]), :][np.array([0,1,3,2,6,7,5,4]), :]
        X_check, Y_check, Z_check = check_cube[:, 0], check_cube[:, 1], check_cube[:, 2]

        distance_check = ((X_check[0] - X_check[1])**2 + (Y_check[0] - Y_check[1])**2 + (Z_check[0] - Z_check[1])**2)**0.5

        distance = ((X_pred[0] - X_pred[1])**2 + (Y_pred[0] - Y_pred[1])**2 + (Z_pred[0] - Z_pred[1])**2)**0.5
        print(distance_check)

        # Scatter original data
        ax.scatter(cube[:, 0], cube[:, 1], cube[:, 2], color='b', linewidth=0.5)

        # Scatter prediction data
        ax.scatter(predicted_cube[:, 0], predicted_cube[:, 1], predicted_cube[:, 2], color='r', linewidth=0.5)

        ax.scatter(check_cube[:, 0], check_cube[:, 1], check_cube[:, 2], color='black', linewidth=0.5)

        ax.plot(cube[:, 0], cube[:, 1], cube[:, 2], label="conv pos == label")
        ax.plot(predicted_cube[:, 0], predicted_cube[:, 1], predicted_cube[:, 2], c="r", label="prediction")
        ax.plot(check_cube[:, 0], check_cube[:, 1], check_cube[:, 2], c="black", label="real pos")

        ax.set_xlim3d(-15, 15)
        ax.set_ylim3d(-15, 15)
        ax.set_zlim3d(0, 40)
        ax.legend()

    # Interval : Delay between frames in milliseconds.
    ani = animation.FuncAnimation(fig, update, 225, interval=100, repeat=False)

    plt.show()


if __name__ == "__main__":
    nr_frames = 225
    data_type = "dual_quat"
    architecture = "fcnn"

    model = load_model(data_type, architecture)

    plot_data, ori_data, pos_data, start = get_random_sim_data(data_type, nr_frames)

    if architecture == "fcnn":
        prediction = get_prediction_fcnn(ori_data, data_type, plot_data, start)
    elif architecture == "lstm":
        prediction = get_prediction_lstm(ori_data, data_type, plot_data, start, 5, out_is_in=False)

    plot_3D_animation(np.array(plot_data), np.array(prediction), np.array(pos_data))
