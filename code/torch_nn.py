import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import torch.utils.data as data
import random
from convert import *
import wandb
import time

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class fcnn(nn.Module):
    def __init__(self, n_data, config):
        super().__init__()

        # Add first layers
        self.layers = [
            nn.Linear(config["n_frames"] * n_data, config["hidden_sizes"][0])
        ]

        # Add consecuative layers with batch_norm / activation funct / dropout
        # as defined in config
        for i in range(len(config["hidden_sizes"])):

            if config["batch_norm"][i]:
                self.layers += [nn.BatchNorm1d(config["hidden_sizes"][i])]

            if config["activation_func"][i] == "Tanh":
                self.layers += [nn.Tanh()]
            elif config["activation_func"][i] == "ReLU":
                self.layers += [nn.ReLU()]
            else:
                raise ValueError("Wrong activation func")

            self.layers += [nn.Dropout(p=config["dropout"][i])]

            if i < len(config["hidden_sizes"]) - 1:
                self.layers += [
                    nn.Linear(config["hidden_sizes"][i], config["hidden_sizes"][i + 1])
                ]

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
        self.pos_target = []
        self.start_pos = []

        # Loop through all simulations
        for i in self.sims:
            with open(f"data/sim_{i}.pickle", "rb") as f:
                data_all = pickle.load(f)["data"]
                # Collect data from data_type
                data = data_all[self.data_type]
                pos_data = data_all["pos"]
                # Add data and targets
                for frame in range(len(data) - (self.n_frames_perentry + 1)):
                    # Always save the start_position for converting
                    if self.data_type == "pos_diff_start":
                        self.start_pos.append(data_all["pos"][0].flatten())
                    else:
                        self.start_pos.append(data_all["start"].flatten())

                    train_end = frame + self.n_frames_perentry

                    self.data.append(data[frame:train_end].flatten())
                    self.target.append(data[train_end + 1].flatten())

                    self.pos_target.append(pos_data[train_end + 1].flatten())

        # TODO CUDA torch.cuda.FloatTensor
        self.data = torch.FloatTensor(np.asarray(self.data))
        self.target = torch.FloatTensor(np.asarray(self.target))
        self.pos_target = torch.FloatTensor(np.asarray(self.pos_target))
        self.start_pos = torch.FloatTensor(np.asarray(self.start_pos))

    def __len__(self):
        # Number of data point we have
        return self.data.shape[0]

    def __getitem__(self, idx):
        # Return the idx-th data point of the dataset
        data_point = self.data[idx]
        data_target = self.target[idx]
        data_start = self.start_pos[idx]
        data_pos_target = self.pos_target[idx]
        return data_point, data_target, data_start, data_pos_target


def train_log(loss, epoch):
    """
    Log the train loss to Weights and Biases
    """
    wandb.log({"Epoch": epoch, "Train loss": loss}, step=epoch)
    print(f"Loss : {loss:.3f}")


def train_model(
    model, optimizer, data_loader, test_loader, loss_module, num_epochs, config
):
    # Set model to train mode
    loss_type = config.loss_type
    model.train()
    wandb.watch(model, loss_module, log="all", log_freq=10)

    # Training loop
    for epoch in range(num_epochs):
        epoch_time = time.time()
        loss_epoch = 0
        for data_inputs, data_labels, start_pos, pos_target in data_loader:
            start = time.time()

            # Set data to current device
            data_inputs = data_inputs.to(device)
            data_labels = data_labels.to(device)
            start_pos = start_pos.to(device)
            pos_target = pos_target.to(device)
            # print("here", pos_target.shape)

            # Get predictions
            preds = model(data_inputs)

            # print("inputs", data_inputs.shape)
            # print("labels", data_labels.shape)
            # print("pos_target", pos_target.shape)

            # Convert predictions to xyz-data
            # conv_time = time.time()
            alt_preds = convert(preds, start_pos, data_loader.dataset.data_type)
            # print("conv_time", time.time() - conv_time)

            # Determine norm penalty for quaternion data
            if config["data_type"] == "quat" or config["data_type"] == "dual_quat":
                norm_penalty = (
                    config["lam"]
                    * (1 - torch.mean(torch.norm(preds[:, :4], dim=-1))) ** 2
                )
            else:
                norm_penalty = 0

            position_loss = loss_module(alt_preds, pos_target)

            # Calculate the total loss
            loss = position_loss + norm_penalty

            loss_epoch += loss

            # Perform backpropagation
            # back_time = time.time()
            optimizer.zero_grad()
            loss.backward()
            # print("backpropagate", time.time() - back_time)

            # Update the parameters
            optimizer.step()
            print("total_time", time.time() - start)

        # Log and print epoch every 10 epochs
        if epoch % 10 == 0:
            # Log to W&B
            train_log(loss_epoch / len(data_loader), epoch)

            # Evaluate model
            true_loss, convert_loss = eval_model(model, test_loader, loss_module, config)

            # Set model to train mode
            model.train()

            print(
                epoch,
                round(loss_epoch.item() / len(data_loader), 10),
                "\t",
                round(convert_loss, 10),
            )

            # Write to file
            f = open(
                f"results/{config.data_type}/{num_epochs}_{config.learning_rate}_{loss_type}.txt",
                "a",
            )
            f.write(
                f"{[epoch, round(loss_epoch.item()/len(data_loader), 10), round(true_loss, 10), round(convert_loss, 10)]} \n"
            )
            f.write("\n")
            f.close()
        print("epoch_time", time.time() - epoch_time)


def eval_model(model, data_loader, loss_module, config):

    model.eval()  # Set model to eval mode

    with torch.no_grad():  # Deactivate gradients for the following code
        total_loss = 0
        total_convert_loss = 0
        for data_inputs, data_labels, start_pos, pos_target in data_loader:

            # Set data to current device
            data_inputs = data_inputs.to(device)
            data_labels = data_labels.to(device)
            # start_pos = start_pos.to(device)
            # pos_target = pos_target.to(device)

            # Get predictions
            preds = model(data_inputs)
            preds = preds.squeeze(dim=1)

            # if config['data_type'] == 'pos':
            #     preds = preds.reshape((preds.shape[0], 8, 3))
            #     data_labels = data_labels.reshape((data_labels.shape[0], 8, 3))

            # Convert predictions to xyz-data
            alt_preds = convert(
                preds.detach().cpu(), start_pos, data_loader.dataset.data_type
            )
            # Determine norm penalty for quaternion data
            if config["data_type"] == "quat" or config["data_type"] == "dual_quat":
                norm_penalty = (
                    config["lam"]
                    * (1 - torch.mean(torch.norm(preds[:, :4], dim=-1))) ** 2
                )
            else:
                norm_penalty = 0

            position_loss = loss_module(alt_preds, pos_target)

            # Calculate the total xyz-loss
            total_convert_loss += position_loss + norm_penalty

            total_loss += loss_module(preds, data_labels)

        # Log loss to W&B
        wandb.log({"Converted test loss": total_convert_loss / len(data_loader)})

    # Return the average loss
    return total_loss.item() / len(data_loader), total_convert_loss.item() / len(
        data_loader
    )


def model_pipeline(hyperparameters, ndata_dict, loss_dict, optimizer_dict):
    # tell wandb to get started
    with wandb.init(project="thesis", config=hyperparameters):
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config

        # make the model, data, and optimization problem
        model, train_loader, test_loader, criterion, optimizer = make(
            config, ndata_dict, loss_dict, optimizer_dict
        )
        print(config["data_type"])
        print(model)

        # and use them to train the model
        train_model(
            model,
            optimizer,
            train_loader,
            test_loader,
            criterion,
            config.epochs,
            config,
        )

        # and test its final performance
        eval_model(model, test_loader, criterion, config)

    return model


def make(config, ndata_dict, loss_dict, optimizer_dict):
    # Make the data
    data_set_train = MyDataset(
        sims=config.train_sims,
        n_frames=config.n_frames,
        n_data=ndata_dict[config.data_type],
        data_type=config.data_type,
    )
    data_set_test = MyDataset(
        sims=config.test_sims,
        n_frames=config.n_frames,
        n_data=ndata_dict[config.data_type],
        data_type=config.data_type,
    )

    train_data_loader = data.DataLoader(
        data_set_train, batch_size=config.batch_size, shuffle=True
    )
    test_data_loader = data.DataLoader(
        data_set_test, batch_size=config.batch_size, shuffle=True, drop_last=False
    )

    # Make the model
    model = fcnn(ndata_dict[config.data_type], config).to(device)

    # Make the loss and optimizer
    criterion = loss_dict[config.loss_type](reduction=config.loss_reduction_type)
    optimizer = optimizer_dict[config.optimizer](
        model.parameters(), lr=config.learning_rate
    )

    return model, train_data_loader, test_data_loader, criterion, optimizer


if __name__ == "__main__":
    n_sims = 2000
    # Divide the train en test dataset
    sims = {i for i in range(n_sims)}
    train_sims = set(random.sample(sims, int(0.8 * n_sims)))
    test_sims = sims - train_sims

    # Set config
    config = dict(
        learning_rate=0.005,
        epochs=5,
        batch_size=128,
        loss_type="L1",
        loss_reduction_type="mean",
        optimizer="Adam",
        data_type="eucl_motion",
        architecture="fcnn",
        train_sims=list(train_sims),
        test_sims=list(test_sims),
        n_frames=10,
        n_sims=n_sims,
        hidden_sizes=[128, 256],
        activation_func=["ReLU", "ReLU"],
        dropout=[0.2, 0.4],
        batch_norm=[True, True, True],
        lam=0.01,
    )

    loss_dict = {"L1": nn.L1Loss, "L2": nn.MSELoss}

    optimizer_dict = {"Adam": torch.optim.Adam}

    ndata_dict = {
        "pos": 24,
        "eucl_motion": 12,
        "quat": 7,
        "log_quat": 7,
        "dual_quat": 8,
        "pos_diff": 24,
        "pos_diff_start": 24,
        "pos_norm": 24,
    }

    start_time = time.time()
    model = model_pipeline(config, ndata_dict, loss_dict, optimizer_dict)
    print("It took ", time.time() - start_time, " seconds.")

    # Save model
    model_dict = {
        "config": config,
        "data_dict": ndata_dict,
        "model": model.state_dict(),
    }

    torch.save(
        model_dict, f"models/{config['data_type']}_{config['architecture']}.pickle"
    )
