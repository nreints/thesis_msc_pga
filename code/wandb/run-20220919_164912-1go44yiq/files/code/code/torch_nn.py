import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import torch.utils.data as data
import random
from convert import *
import wandb

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class Network(nn.Module):

    def __init__(self, n_data, config):
        super().__init__()

        # Add first layes
        self.layers = [nn.Linear(config["n_frames"] * n_data, config["hidden_sizes"][0])]

        # Add consecuative layers with batch_norm / activation funct / dropout
            # As defined in config
        for i in range(len(config["hidden_sizes"])):

            if config["batch_norm"][i]:
                self.layers += [nn.BatchNorm1d(config["hidden_sizes"][i])]

            if config['activation_func'][i] == "Tanh":
                self.layers += [nn.Tanh()]
            elif config["activation_func"][i] == "ReLU":
                self.layers += [nn.ReLU()]
            else:
                raise ValueError('Wrong activation func')

            self.layers += [nn.Dropout(p=config["dropout"][i])]

            if i < len(config["hidden_sizes"]) - 1:
                self.layers += [nn.Linear(config["hidden_sizes"][i], config["hidden_sizes"][i+1])]

        self.layers += [nn.Linear(config["hidden_sizes"][-1], n_data)]

        self.linears = nn.Sequential(*self.layers)

    def forward(self, x):
        # Perform the calculation of the model to determine the prediction
        return self.linears(x)



class MyDataset(data.Dataset):

    def __init__(self, sims, n_frames, n_data, data_type):
        """
        Inputs:
            n_sims -
            size - Number of data points we want to generate
            std - Standard deviation of the noise (see generate_continuous_xor function)
        """
        super().__init__()
        self.n_frames_perentry = n_frames
        self.n_datap_perframe = n_data
        self.sims = sims
        self.data_type = data_type
        self.collect_data()

    def collect_data(self):

        self.data = []
        self.target = []
        self.start_pos = []

        for i in self.sims:
            with open(f'data/sim_{i}.pickle', 'rb') as f:
                data_all = pickle.load(f)["data"]
                # Collect data from data_type
                data = data_all[self.data_type]
                # Add data and targets
                for frame in range(len(data) - (self.n_frames_perentry + 1)):
                    # Always save the start_position for converting
                    self.start_pos.append(data_all["pos"][0])
                    train_end = frame + self.n_frames_perentry
                    self.data.append(data[frame:train_end].flatten())
                    self.target.append(data[train_end+1].flatten())

        self.data = torch.FloatTensor(np.asarray(self.data))
        self.target = torch.FloatTensor(np.asarray(self.target))
        self.start_pos = torch.FloatTensor(np.asarray(self.start_pos))

    def __len__(self):
        # Number of data point we have. Alternatively self.data.shape[0], or self.label.shape[0]
        return self.data.shape[0]

    def __getitem__(self, idx):
        # Return the idx-th data point of the dataset
        # If we have multiple things to return (data point and label), we can return them as tuple
        data_point = self.data[idx]
        data_target = self.target[idx]
        data_start = self.start_pos[idx]
        return data_point, data_target, data_start


def train_log(loss, epoch):
    """
    Log the train loss to Weights and Biases
    """
    wandb.log({"Epoch": epoch, "Train loss": loss}, step=epoch)
    print(f"Loss after " + f" examples: {loss:.3f}")

def train_model(model, optimizer, data_loader, test_loader, loss_module, num_epochs, config):
    # Set model to train mode
    loss_type = config.loss_type
    model.train()
    wandb.watch(model, loss_module, log="all", log_freq=10)

    # Training loop
    for epoch in range(num_epochs):
        loss_epoch = 0
        for data_inputs, data_labels, start_pos in data_loader:
            ## Step 1: Move input data to device (only strictly necessary if we use GPU)
            data_inputs = data_inputs.to(device)
            # print(data_inputs)

            data_labels = data_labels.to(device)
            # print(data_labels)
            start_pos = start_pos.to(device)

            ## Step 2: Run the model on the input data
            preds = model(data_inputs)
            preds = preds.squeeze(dim=1) # Output is [Batch size, 1], but we want [Batch size]

            ## Step 3: Calculate the loss
            alt_preds = convert(preds, start_pos, data_loader.dataset.data_type)
            alt_labels = convert(data_labels, start_pos, data_loader.dataset.data_type)

            loss = loss_module(alt_preds, alt_labels)
            # loss = loss_module(preds, data_labels)

            loss_epoch += loss

            # Perform backpropagation
            optimizer.zero_grad()
            loss.backward()

            # Update the parameters
            optimizer.step()

        # Log and print epoch every 10 epochs
        if epoch % 10 == 0:
            # Log to W&B
            train_log(loss_epoch/len(data_loader), epoch)

            # Evaluate model
            true_loss, convert_loss = eval_model(model, test_loader, loss_module)

            # Set model to train mode
            model.train()

            print(epoch, round(loss_epoch.item()/len(data_loader), 10), "\t", round(convert_loss, 10))

            # Write to file
            f = open(f"results/{config.data_type}/{num_epochs}_{config.learning_rate}_{loss_type}.txt", "a")
            f.write(f"{[epoch, round(loss_epoch.item()/len(data_loader), 10), round(true_loss, 10), round(convert_loss, 10)]} \n")
            f.write("\n")
            f.close()


    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # Axes3D.scatter(xs, ys, zs=0, zdir='z', s=20, c=None, depthshade=True, *args, **kwargs)



def eval_model(model, data_loader, loss_module):
    model.eval() # Set model to eval mode

    with torch.no_grad(): # Deactivate gradients for the following code
        total_loss = 0
        total_convert_loss = 0
        for data_inputs, data_labels, start_pos in data_loader:

            # Determine prediction of model on dev set
            data_inputs, data_labels = data_inputs.to(device), data_labels.to(device)
            preds = model(data_inputs)
            preds = preds.squeeze(dim=1)

            alt_preds = convert(preds.detach().cpu(), start_pos, data_loader.dataset.data_type)
            alt_labels = convert(data_labels.detach().cpu(), start_pos, data_loader.dataset.data_type)

            total_loss += loss_module(preds, data_labels)
            total_convert_loss += loss_module(alt_preds, alt_labels)

        # Log loss to Weights and Biases
        wandb.log({"Converted test loss": total_convert_loss/len(data_loader)})

    # Return the average loss
    return total_loss.item()/len(data_loader), total_convert_loss.item()/len(data_loader)




n_sims = 750
# Divide the train en test dataset
sims = {i for i in range(n_sims)}
train_sims = set(random.sample(sims, int(0.8 * n_sims)))
test_sims = sims - train_sims



def model_pipeline(hyperparameters, ndata_dict, loss_dict, optimizer_dict):
    # tell wandb to get started
    with wandb.init(project="thesis", config=hyperparameters):
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config

        # make the model, data, and optimization problem
        model, train_loader, test_loader, criterion, optimizer = make(config, ndata_dict, loss_dict, optimizer_dict)
        print(model)

        # and use them to train the model
            #   model, optimizer, data_loader, test_loader, loss_module, num_epochs=100, loss_type="L1"):
        train_model(model, optimizer, train_loader, test_loader, criterion, config.epochs, config)

        # and test its final performance
        eval_model(model, test_loader, criterion)

    return model

def make(config, ndata_dict, loss_dict, optimizer_dict):
    # Make the data
    data_set_train = MyDataset(sims=config.train_sims, n_frames=config.n_frames, n_data=ndata_dict[config.data_type], data_type=config.data_type)
    data_set_test = MyDataset(sims=config.test_sims, n_frames=config.n_frames, n_data=ndata_dict[config.data_type], data_type=config.data_type)

    train_data_loader = data.DataLoader(data_set_train, batch_size=config.batch_size, shuffle=True)
    test_data_loader = data.DataLoader(data_set_test, batch_size=config.batch_size, shuffle=True, drop_last=False)

    # Make the model
    model = Network(ndata_dict[config.data_type], config).to(device)

    # Make the loss and optimizer
    criterion = loss_dict[config.loss_type](reduction=config.loss_reduction_type)
    optimizer = optimizer_dict[config.optimizer](
        model.parameters(), lr=config.learning_rate)

    return model, train_data_loader, test_data_loader, criterion, optimizer

config = dict(
    learning_rate = 0.1,
    epochs = 10,
    batch_size = 128,
    loss_type = "L1",
    loss_reduction_type = "mean",
    optimizer = "Adam",
    data_type = "pos",
    architecture = "fcnn",
    train_sims = list(train_sims),
    test_sims = list(test_sims),
    n_frames = 20,
    n_sims = n_sims,
    hidden_sizes = [128, 256, 128],
    activation_func = ["Tanh", "Tanh", "ReLU"],
    dropout = [0.4, 0.2, 0.3],
    batch_norm = [True, True, True]
)

loss_dict = {'L1': nn.L1Loss,
                'L2': nn.MSELoss}

optimizer_dict = {'Adam': torch.optim.Adam}

ndata_dict = {"pos": 24,
                "eucl_motion": 12,
                "quat": 7,
                "log_quat": 7,
                "pos_diff": 24,
                "pos_diff_start": 24,
                "pos_norm": 24
            }

model = model_pipeline(config, ndata_dict, loss_dict, optimizer_dict)
model_dict = {'config': config,
            'data_dict':ndata_dict,
            'model': model.state_dict()}
torch.save(model_dict, f"models/{config['data_type']}_{config['architecture']}.pickle")