from platform import architecture
import torch
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from torch_nn import Network
import pickle
from random import randint
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from convert import *


def load_model(data_type, architecture):
    # Load model
    model_dict = torch.load(f"models/{data_type}_{architecture}.pickle")
    config = model_dict['config']
    ndata_dict = model_dict['data_dict']

    model = Network(ndata_dict[config['data_type']], config)
    model.load_state_dict(model_dict['model'])
    model.eval()
    print("Current model: \n", model)

    return model

def get_random_sim_data(data_type, nr_frames):
    # Select random simulation
    i = randint(0, 749)

    i=10
    print("Using simulation number ", i)
    with open(f'data/sim_{i}.pickle', 'rb') as f:
        file = pickle.load(f)

        start_pos = torch.tensor(file["data"]["pos"][0], dtype=torch.float32)
        start_pos = start_pos[None, :].repeat(nr_frames, 1, 1)

        data_tensor = torch.tensor(file["data"][data_type], dtype=torch.float32)

        # Get data as data_type
        original_data = data_tensor.flatten(start_dim=1)

        # Convert to pos data for plotting
        plot_data = convert(data_tensor.flatten(start_dim=1), start_pos, data_type).reshape(nr_frames, 8, 3)

    #     print("--- PLOT ----")
    #     # Check if converting went correctly
    #     print("logQuat", original_data[0])
    #     print("start", start_pos[0])
    #     print("xyz", plot_data[0])
    # exit()
    return plot_data, original_data

def get_prediction(original_data, data_type, xyz_data):
    # Collect prediction of model given simulation
    # Result should be xyz data for plot
    result = torch.zeros_like(xyz_data)

    # Get first position
    start_pos = xyz_data[0][None, :]

    for frame_id in range(20, xyz_data.shape[0]):
        # Get 20 frames shape: (1, 480)
        input_data = original_data[frame_id - 20 : frame_id]

        input_data = input_data.flatten()[None, :]

        # Save the prediction in result
        with torch.no_grad(): # Deactivate gradients for the following code
            prediction = model(input_data)
            print(convert(prediction, start_pos, data_type))
            # print(xyz_data[frame_id])


            result[frame_id] = convert(prediction, start_pos, data_type)

        # if frame_id == 35:
        #     exit()

    return result

def plot_3D_animation(data, result):
    data = data.numpy()
    result = result.numpy()

    # Open figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Set the axis limits
    ax.set_xlim3d(-15, 15)
    ax.set_ylim(-15, 15)
    ax.set_zlim(0, 50)
    ax.set_xlabel('$X$')
    ax.set_ylabel('$Y$')
    ax.set_xlabel('$Z$')

    # Initial plot
    first_cube = data[0]
    first_cube = first_cube[np.array([0, 1, 2, 3, 4, 5, 6, 7]), :][np.array([0,1,3,2,6,7,5,4]), :]

    cube_result = result[0]
    cube_result = cube_result[np.array([0, 1, 2, 3, 4, 5, 6, 7]), :][np.array([0,1,3,2,6,7,5,4]), :]

    X, Y, Z = first_cube[:, 0], first_cube[:, 1], first_cube[:, 2]
    X_pred, Y_pred, Z_pred = cube_result[:, 0], cube_result[:, 1], cube_result[:, 2]

    # Begin plotting.
    ax.scatter(X, Y, Z, linewidth=0.5, color='b')
    ax.scatter(X_pred, Y_pred, Z_pred, color='r', linewidth=0.5)
    ax.plot(X, Y, Z)
    ax.plot(X_pred, Y_pred, Z_pred, c="r")


    def update(idx):
        ax.set_xlabel('$X$')
        ax.set_ylabel('$Y$')
        ax.set_xlabel('$Z$')

        ax.set_xlim3d(-15, 15)
        ax.set_ylim3d(-15, 15)
        ax.set_zlim3d(0, 50)


        # Remove the previous scatter plot
        if idx != 0:
            ax.cla()

        # Get original cube data
        cube = data[idx]
        cube = cube[np.array([0, 1, 2, 3, 4, 5, 6, 7]), :][np.array([0,1,3,2,6,7,5,4]), :]

        # Get predicted cube date
        predicted_cube = result[idx]
        predicted_cube = predicted_cube[np.array([0, 1, 2, 3, 4, 5, 6, 7]), :][np.array([0,1,3,2,6,7,5,4]), :]


        # Scatter original data
        ax.scatter(cube[:, 0], cube[:, 1], cube[:, 2], color='b', linewidth=0.5)

        # Scatter prediction data
        ax.scatter(predicted_cube[:, 0], predicted_cube[:, 1], predicted_cube[:, 2], color='r', linewidth=0.5)

        ax.plot(cube[:, 0], cube[:, 1], cube[:, 2])
        ax.plot(predicted_cube[:, 0], predicted_cube[:, 1], predicted_cube[:, 2], c="r")


    # Interval : Delay between frames in milliseconds.
    ani = animation.FuncAnimation(fig, update, 225, interval=100, repeat=False)

    plt.show()


if __name__ == "__main__":
    nr_frames = 225
    data_type = "log_quat"
    architecture = "fcnn"

    model = load_model(data_type, architecture)

    plot_data, ori_data = get_random_sim_data(data_type, nr_frames)

    prediction = get_prediction(ori_data, data_type, plot_data)

    plot_3D_animation(plot_data, prediction)
