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
import os
import argparse
from pathlib import Path

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
wandb.login(key="dc4407c06f6d57a37befe29cb0773deffd670c72")

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
    def __init__(self, sims, n_frames, n_data, data_type, dir):
        """
        Inputs:
            n_sims - Number of simulations.
            size - Number of data points we want to generate
            std - Standard deviation of the noise (see generate_continuous_xor function)
        """
        super().__init__()
        self.n_frames_perentry = n_frames
        self.n_datap_perframe = n_data
        self.sims = sims
        self.data_type = data_type
        self.dir = dir
        self.collect_data()

    def collect_data(self):

        self.data = []
        self.target = []
        self.pos_target = []
        self.start_pos = []

        # Loop through all simulations
        for i in self.sims:
            with open(f"{self.dir}/sim_{i}.pickle", "rb") as f:
                data_all = pickle.load(f)["data"]
                # Collect data from data_type
                data = data_all[self.data_type]
                pos_data = data_all["pos"]
                # Add data and targets
                for frame in range(len(data) - (self.n_frames_perentry + 1)):
                    # Always save the start position for converting
                    if self.data_type == "pos_diff_start":
                        self.start_pos.append(data_all["pos"][0].flatten())
                    else:
                        self.start_pos.append(data_all["start"].flatten())

                    train_end = frame + self.n_frames_perentry

                    self.data.append(data[frame:train_end].flatten())
                    self.target.append(data[train_end + 1].flatten())

                    self.pos_target.append(pos_data[train_end + 1].flatten())

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

def train_model(
    model, optimizer, data_loader, test_loaders, loss_module, num_epochs, config, losses
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

            # Get predictions
            preds = model(data_inputs)

            # conv_time = time.time()
            # Convert predictions to xyz-data
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
            # print("total_time", time.time() - start)

        # Log to W&B
        train_log(loss_epoch / len(data_loader), epoch)

        # Evaluate model
        true_loss, convert_loss, total_convert_loss = eval_model(model, test_loaders, loss_module, config, epoch, losses)

        # Set model to train mode
        model.train()

        print(
            epoch,
            round(loss_epoch.item() / len(data_loader), 10),
            "\t",
            round(convert_loss, 10), "\t", round(total_convert_loss, 10)
        )

        # Write to file
    # f = open(
        #     f"results/{config.data_type}/{num_epochs}_{config.learning_rate}_{loss_type}.txt",
        #     "a",
        # )
        # f.write(
        #     f"{[epoch, round(loss_epoch.item()/len(data_loader), 10), round(true_loss, 10), round(convert_loss, 10)]} \n"
        # )
        # f.write("\n")
        # f.close()
        print("epoch_time; ", time.time() - epoch_time)

def eval_model(model, data_loaders, loss_module, config, current_epoch, losses):

    model.eval()  # Set model to eval mode

    with torch.no_grad():  # Deactivate gradients for the following code
        for i, data_loader in enumerate(data_loaders):
            for loss_module in losses:
                loss_module = loss_module(reduction=config.loss_reduction_type)
                total_loss = 0
                total_convert_loss = 0
                wandb_total_convert_loss = 0
                for data_inputs, data_labels, start_pos, pos_target in data_loader:

                    # Set data to current device
                    data_inputs = data_inputs.to(device)
                    data_labels = data_labels.to(device)
                    # start_pos = start_pos.to(device)
                    # pos_target = pos_target.to(device)

                    # Get predictions
                    preds = model(data_inputs)
                    preds = preds.squeeze(dim=1)

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
                    wandb_total_convert_loss += position_loss

                    total_loss += loss_module(preds, data_labels)

                # Log loss to W&B
                print(f"\t Logging test loss: {config.data_dirs_test[i][5:]} => {wandb_total_convert_loss / len(data_loader)}")
                wandb.log({f"Test loss {config.data_dirs_test[i][5:]}": wandb_total_convert_loss / len(data_loader)}, step=current_epoch)
                # wandb.log({f"Test loss {config.data_dirs_test[i][5:]}": wandb_total_convert_loss / len(data_loader)})

    # Return the average loss
    return total_loss.item() / len(data_loader), wandb_total_convert_loss.item() / len(
        data_loader), total_convert_loss.item() / len(
        data_loader)

def model_pipeline(hyperparameters, ndata_dict, loss_dict, optimizer_dict, mode_wandb, losses):
    # tell wandb to get started
    with wandb.init(project="thesis", config=hyperparameters, mode=mode_wandb, tags=[device]):
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config
        wandb.run.name = f"{config.architecture}/{config.data_type}/{config.iter}"

        # make the model, data, and optimization problem
        model, train_loader, test_loaders, criterion, optimizer = make(
            config, ndata_dict, loss_dict, optimizer_dict,
        )
        print(config["data_type"])
        print(model)

        # and use them to train the model
        train_model(
            model,
            optimizer,
            train_loader,
            test_loaders,
            criterion,
            config.epochs,
            config,
        )

        # and test its final performance
        eval_model(model, test_loaders, criterion, config, config.epochs, losses)

    return model


def make(config, ndata_dict, loss_dict, optimizer_dict):
    # Make the data
    data_set_train = MyDataset(
        sims=config.train_sims,
        n_frames=config.n_frames,
        n_data=ndata_dict[config.data_type],
        data_type=config.data_type,
        dir=config.data_dir_train
    )
    train_data_loader = data.DataLoader(
        data_set_train, batch_size=config.batch_size, shuffle=True
    )
    test_data_loaders = []

    for test_data_dir in config.data_dirs_test:
        # print("data/"+test_data_dir)
        data_set_test = MyDataset(
            sims=config.test_sims,
            n_frames=config.n_frames,
            n_data=ndata_dict[config.data_type],
            data_type=config.data_type,
            dir="data/"+test_data_dir
        )
        test_data_loader = data.DataLoader(
            data_set_test, batch_size=config.batch_size, shuffle=True, drop_last=False
        )
        test_data_loaders += [test_data_loader]


    # Make the model
    model = fcnn(ndata_dict[config.data_type], config).to(device)

    # Make the loss and optimizer
    criterion = loss_dict[config.loss_type](reduction=config.loss_reduction_type)
    optimizer = optimizer_dict[config.optimizer](
        model.parameters(), lr=config.learning_rate
    )

    return model, train_data_loader, test_data_loaders, criterion, optimizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("-n_sims", type=int, help="number of simulations", default=5000)
    # parser.add_argument("-n_frames", type=int, help="number of frames", default=1000)
    parser.add_argument("-mode_wandb", type=str, help="mode of wandb: online, offline, disabled", default="online")
    parser.add_argument("-data_dir_train", type=str, help="directory of the train data", nargs="+", default="data_t(0, 0)_r(0, 0)_none")
    parser.add_argument("-loss", type=str, help="Loss type", default="L2")
    # parser.add_argument("-data_dir_test", type=list, help="directory of the test data", default="")
    parser.add_argument("-data_type", type=str, help="Type of data", default="pos")
    parser.add_argument("-iterations", type=int, help="Number of iterations", default=1)
    args = parser.parse_args()

    data_dir_train = "data/" + " ".join(args.data_dir_train)
    # data_dirs_test = args.data_dir_test
    data_dirs_test = ["data_t(0, 0)_r(0, 0)_none", "data_t(-10, 10)_r(0, 0)_none",
                        "data_t(0, 0)_r(-5, 5)_none","data_t(-10, 10)_r(-5, 5)_none"]
    # if args.data_dir_test == "":
    #     data_dirs_test = [data_dir_train]
    # else:
    #     data_dirs_test = "data/" + args.data_dir_test

    losses = [nn.MSELoss, nn.L1Loss]

    if not os.path.exists(data_dir_train):
        raise IndexError("No directory for the train data {args.data_dir_train}")

    for i in range(args.iterations):
        # Divide the train en test dataset
        n_sims_train = len(os.listdir(data_dir_train))
        sims_train = {i for i in range(n_sims_train)}
        train_sims = set(random.sample(sims_train, int(0.8 * n_sims_train)))
        test_sims = sims_train - train_sims

        # if data_dir_train == data_dirs_test:
        #     train_sims = set(random.sample(sims_train, int(0.8 * n_sims_train)))
        #     test_sims = sims_train - train_sims
        # else:
        #     train_sims = sims_train
        #     n_sims_test = len(os.listdir(data_dir_test))
        #     # Use maximum number of test simulations or 20% of the train simulations
        #     if n_sims_test < int(n_sims_train * 0.2):
        #         print(f"Less than 20% of number train sims as test sims.")
        #         test_sims = {i for i in range(n_sims_test)}
        #     else:
        #         test_sims = set(random.sample(sims_train, int(0.2 * n_sims_test)))



        print(f"Number of train simulations: {len(train_sims)}")
        print(f"Number of test simulations: {len(test_sims)}")
        # Set config
        config = dict(
            learning_rate=0.001,
            epochs=30,
            batch_size=1024,
            loss_type=args.loss,
            loss_reduction_type="mean",
            optimizer="Adam",
            data_type=args.data_type,
            architecture="fcnn",
            train_sims=list(train_sims),
            test_sims=list(test_sims),
            n_frames=10,
            n_sims=n_sims_train,
            hidden_sizes=[128, 256],
            activation_func=["ReLU", "ReLU"],
            dropout=[0.2, 0.4],
            batch_norm=[True, True, True],
            lam=0.01,
            data_dir_train=data_dir_train,
            data_dirs_test=data_dirs_test,
            iter=i
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
            "log_dualQ": 6
        }

        start_time = time.time()
        model = model_pipeline(config, ndata_dict, loss_dict, optimizer_dict, args.mode_wandb, losses)
        print(f"It took {time.time() - start_time} seconds to train & eval the model.")

        # Save model
        model_dict = {
            "config": config,
            "data_dict": ndata_dict,
            "model": model.state_dict(),
        }
        if not os.path.exists("models"):
            os.mkdir("models")

        torch.save(
            model_dict, f"models/fcnn/{config['data_type']}_{config['architecture']}_{args.data_dir_train}.pickle"
        )
