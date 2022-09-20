from platform import architecture
import torch
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from torch_nn import Network
import pickle
from random import randint

import matplotlib.pyplot as plt
import matplotlib.animation as animation

# model = torch.load(f"models/pos_fcnn.pickle")
# print(model)

data_type = "pos"
architecture = "fcnn"

model_dict = torch.load(f"models/{data_type}_{architecture}.pickle")
config = model_dict['config']
ndata_dict = model_dict['data_dict']
model = Network(ndata_dict[config['data_type']], config)
model.load_state_dict(model_dict['model'])
model.eval()
print(model)
# exit()

i = randint(0, 749)
print(i)
with open(f'data/sim_{i}.pickle', 'rb') as f:
    data = torch.FloatTensor(pickle.load(f)["data"][data_type])

result = torch.zeros_like(data)
for frame_id in range(20, data.shape[0]):
    # Get 20 frames (1, 480)
    input_data = data[frame_id - 20 : frame_id]
    input_data = input_data.flatten()[None, :]

    # Save the prediction in result
    print(input_data)
    with torch.no_grad(): # Deactivate gradients for the following code
        result[frame_id] = model(input_data).reshape(8, 3)

# result = result.detach(
print("data_shape ", data.shape)
print("result_shape ", result.shape)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


first_cube = data[0]
cube_result = result[0]
X, Y, Z = first_cube[:, 0], first_cube[:, 1], first_cube[:, 2]
X_result, Y_result, Z_result = cube_result[:, 0], cube_result[:, 1], cube_result[:, 2]
# print(X)
# print(Y)
# print(Z)
# exit()
# print(X.shape, Y.shape, Z.shape)

# Set the axis limits
ax.set_xlim3d(-15, 15)
ax.set_ylim(-15, 15)
ax.set_zlim(0, 50)

# Begin plotting.
ax.scatter(X, Y, Z, color='b', linewidth=0.5)
ax.scatter(X_result, Y_result, Z_result, color='r', linewidth=0.5)

ax.set_xlabel('$X$', fontsize=20)
ax.set_ylabel('$Y$')

# plt.show()

def update(idx):
    # Set the axis limits
    print(idx)
    # Remove the previous scatter plot
    if idx != 0:
        ax.cla()

    # Plot the new wireframe and pause briefly before continuing.
    cube = data[idx]
    result_cube = result[idx]
    print(f'prediction: {result_cube[:,0]}')
    print(f'true: {cube[:,0]}')
    ax.scatter(cube[:, 0], cube[:, 1], cube[:, 2], color='b', linewidth=0.5)
    ax.scatter(result_cube[:, 0], result_cube[:, 1], result_cube[:, 2], color='r', linewidth=0.5)
    if idx == 224:
        print(torch.min(result_cube[:, 2]))

# Interval : Delay between frames in milliseconds.

ani = animation.FuncAnimation(fig, update, 225, interval=100, repeat=False)

plt.show()