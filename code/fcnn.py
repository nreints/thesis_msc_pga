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

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class fcnn(nn.Module):
    def __init__(self, n_data, config):
        super().__init__()
        extra_input = config["extra_input_n"]

        # Add first layers
        self.layers = [
            nn.Linear(
                config["n_frames"] * n_data + extra_input, config["hidden_sizes"][0]
            )
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
    def __init__(
        self,
        sims,
        n_frames,
        n_data,
        data_type,
        dir,
        extra_input,
        normalize_extra_input=True,
        return_normalization=True,
    ):
        super().__init__()
        self.n_frames_perentry = n_frames
        self.n_datap_perframe = n_data
        self.sims = sims
        self.data_type = data_type
        self.dir = dir
        self.extra_input = extra_input
        # self.norm_extra_input = normalize_extra_input
        # self.return_normalization = return_normalization
        self.collect_data()

    def collect_data(self):
        start_time = time.time()
        count = 0
        # Loop through all simulations
        for i in self.sims:
            with open(f"{self.dir}/sim_{i}.pickle", "rb") as f:
                data_all = pickle.load(f)["data"]
                # Collect data from data_type
                data = torch.FloatTensor(data_all[self.data_type])
                pos_data = torch.FloatTensor(data_all["pos"])
                # Add data and targets
                if count == 0:
                    data_per_sim = len(data) - (self.n_frames_perentry + 1)
                    len_data = len(self.sims) * data_per_sim
                    self.data = torch.zeros(
                        len_data,
                        self.n_frames_perentry * self.n_datap_perframe
                        + self.extra_input[1],
                    )
                    self.extra_input_data = torch.zeros(len_data, self.extra_input[1])
                    self.target = torch.zeros((len_data, self.n_datap_perframe))
                    self.target_pos = torch.zeros((len_data, 24))
                    self.start_pos = torch.zeros_like(self.target_pos)
                for frame in range(data_per_sim):
                    # Always save the start position for converting
                    self.start_pos[count] = pos_data[0].flatten()
                    train_end = frame + self.n_frames_perentry
                    if self.extra_input[1] != 0:
                        extra_input_values = torch.FloatTensor(
                            data_all[self.extra_input[0]]
                        )
                        self.data[count, -self.extra_input[1] :] = extra_input_values
                        self.data[count, : -self.extra_input[1]] = data[
                            frame:train_end
                        ].flatten()
                    else:
                        self.data[count] = data[frame:train_end].flatten()
                    self.target[count] = data[train_end + 1].flatten()

                    self.target_pos[count] = pos_data[train_end + 1].flatten()
                    count += 1

        self.mean = torch.mean(self.data)
        self.std = torch.std(self.data)
        self.normalized_data = (self.data - self.mean) / self.std

        self.normalize_extra_input = torch.mean(
            torch.norm(self.data[:, -self.extra_input[1] :], dim=1)
        )
        # print("mean of norm extra_input", self.normalize_extra_input.item())
        # self.data[:, -self.extra_input[1] :] = self.extra_input_data
        print(f"The dataloader took {time.time() - start_time} seconds.")

    def __len__(self):
        # Number of data point we have
        return self.data.shape[0]

    def __getitem__(self, idx):
        # Return the idx-th data point of the dataset
        data_point = self.data[idx]
        data_target = self.target[idx]
        data_start = self.start_pos[idx]
        data_pos_target = self.target_pos[idx]
        data_normalized = self.normalized_data[idx]
        return data_point, data_target, data_start, data_pos_target, data_normalized


def train_log(loss, epoch, config):
    """
    Log the train loss to Weights and Biases
    """
    wandb.log({f"Train loss {config.data_dir_train[5:-12]}": loss}, step=epoch)


def train_model(
    model,
    optimizer,
    data_loader,
    test_loaders,
    loss_module,
    num_epochs,
    config,
    losses,
    data_set_train,
):
    print("--- Started Training ---")
    # Set model to train mode
    model.train()
    wandb.watch(model, loss_module, log="all", log_freq=10)

    # Training loop
    for epoch in range(num_epochs):
        epoch_time = time.time()
        loss_epoch = 0
        for data_inputs, data_labels, start_pos, pos_target, data_norm in data_loader:

            # Set data to current device
            data_inputs = data_inputs.to(
                device
            )  # Shape: [batch, frames x n_data + config["extra_input_n"]]
            data_norm = data_norm.to(device)
            data_labels = data_labels.to(device)  # Shape: [batch, n_data]
            pos_target = pos_target.to(device)  # Shape: [batch, n_data]
            start_pos = start_pos.to(device)  # Shape: [batch, n_data]
            if config["str_extra_input"] == "inertia_body":
                data_inputs[:, -config["extra_input_n"] :] = (
                    data_inputs[:, -config["extra_input_n"] :]
                    / data_set_train.normalize_extra_input
                )
            # Get predictions
            preds = model(data_inputs)  # Shape: [batch, n_data]
            # print("perdictions:", preds[0])
            # print("labels", data_labels[0])
            # preds = model(data_norm)
            # preds = preds * data_set_train.std + data_set_train.mean

            # Convert predictions to xyz-data
            alt_preds = convert(preds, start_pos, data_loader.dataset.data_type)
            # print("alt_preds:", alt_preds[0])
            # print("pos_targ", pos_target[0])
            # exit()

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
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

        print(f"Epoch {epoch}")
        # Log to W&B
        train_log(loss_epoch / len(data_loader), epoch, config)
        print(
            f"\t Logging train Loss: {round(loss_epoch.item() / len(data_loader), 10)} ({loss_module}: {config.data_dir_train[5:]})"
        )

        # Evaluate model
        true_loss, convert_loss, total_convert_loss = eval_model(
            model, test_loaders, loss_module, config, epoch, losses, data_set_train
        )

        # Set model to train mode
        model.train()
        print(f"     --> Epoch_time; {time.time() - epoch_time}")


def eval_model(
    model, data_loaders, loss_module, config, current_epoch, losses, data_set_train
):

    model.eval()  # Set model to eval mode

    with torch.no_grad():  # Deactivate gradients for the following code
        for i, data_loader in enumerate(data_loaders):
            for loss_module in losses:
                loss_module = loss_module(reduction=config.loss_reduction_type)
                total_loss = 0
                total_convert_loss = 0
                wandb_total_convert_loss = 0
                for data_inputs, data_labels, start_pos, pos_target, _ in data_loader:
                    if config["str_extra_input"] == "inertia_body":
                        data_inputs[:, -config["extra_input_n"] :] = (
                            data_inputs[:, -config["extra_input_n"] :]
                            / data_set_train.normalize_extra_input
                        )
                    # Set data to current device
                    # data_inputs = data_inputs.to(device)
                    # data_norm = (data_inputs - data_set_train.mean) / data_set_train.std
                    data_labels = data_labels.to(device)

                    # Get predictions
                    preds = model(data_inputs)
                    # preds = model(data_norm)
                    preds = preds.squeeze(dim=1)
                    # preds = preds * data_set_train.std + data_set_train.mean

                    # Convert predictions to xyz-data
                    alt_preds = convert(
                        preds.detach().cpu(), start_pos, data_loader.dataset.data_type
                    )

                    # Determine norm penalty for quaternion data
                    if (
                        config["data_type"] == "quat"
                        or config["data_type"] == "dual_quat"
                    ):
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
                print(
                    f"\t Logging test loss {wandb_total_convert_loss / len(data_loader)} ({loss_module}: {config.data_dirs_test[i][5:]})"
                )
                name = config.data_dirs_test[i][5:-12]
                if name[-1] == "_":
                    name = config.data_dirs_test[i][-12:]
                wandb.log(
                    {
                        f"Test loss {name} {loss_module}": wandb_total_convert_loss
                        / len(data_loader)
                    },
                    step=current_epoch,
                )
                # wandb.log({f"Test loss {config.data_dirs_test[i][5:]}": wandb_total_convert_loss / len(data_loader)})

    # Return the average loss
    return (
        total_loss.item() / len(data_loader),
        wandb_total_convert_loss.item() / len(data_loader),
        total_convert_loss.item() / len(data_loader),
    )


def model_pipeline(
    hyperparameters, ndata_dict, loss_dict, optimizer_dict, mode_wandb, losses
):
    # tell wandb to get started
    with wandb.init(
        project="test", config=hyperparameters, mode=mode_wandb, tags=[device.type]
    ):
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config
        wandb.run.name = f"{config.architecture}/{config.data_type}/{config.iter}/{config.str_extra_input}/"

        # make the model, data, and optimization problem
        model, train_loader, test_loaders, criterion, optimizer, data_set_train = make(
            config,
            ndata_dict,
            loss_dict,
            optimizer_dict,
        )
        print("Datatype:", config["data_type"])

        # and use them to train the model
        train_model(
            model,
            optimizer,
            train_loader,
            test_loaders,
            criterion,
            config.epochs,
            config,
            losses,
            data_set_train,
        )

        # and test its final performance
        eval_model(
            model,
            test_loaders,
            criterion,
            config,
            config.epochs,
            losses,
            data_set_train,
        )

    return model


def make(config, ndata_dict, loss_dict, optimizer_dict):
    # Make the data
    data_set_train = MyDataset(
        sims=config.train_sims,
        n_frames=config.n_frames,
        n_data=ndata_dict[config.data_type],
        data_type=config.data_type,
        dir=config.data_dir_train,
        extra_input=(config.str_extra_input, config.extra_input_n),
        return_normalization=True,
    )
    train_data_loader = data.DataLoader(
        data_set_train, batch_size=config.batch_size, shuffle=True
    )
    print("-- Finished Train Dataloader --")

    test_data_loaders = []

    for test_data_dir in config.data_dirs_test:
        # print("data/"+test_data_dir)
        data_set_test = MyDataset(
            sims=config.test_sims,
            n_frames=config.n_frames,
            n_data=ndata_dict[config.data_type],
            data_type=config.data_type,
            dir="data/" + test_data_dir,
            extra_input=(config.str_extra_input, config.extra_input_n),
        )
        test_data_loader = data.DataLoader(
            data_set_test, batch_size=config.batch_size, shuffle=True, drop_last=False
        )
        test_data_loaders += [test_data_loader]

    print("-- Finished Test Dataloader(s) --")

    # Make the model
    model = fcnn(ndata_dict[config.data_type], config).to(device)

    # Make the loss and optimizer
    criterion = loss_dict[config.loss_type](reduction=config.loss_reduction_type)
    optimizer = optimizer_dict[config.optimizer](
        model.parameters(), lr=config.learning_rate
    )

    return (
        model,
        train_data_loader,
        test_data_loaders,
        criterion,
        optimizer,
        data_set_train,
    )


if __name__ == "__main__":
    wandb.login(key="dc4407c06f6d57a37befe29cb0773deffd670c72")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--mode_wandb",
        type=str,
        choices=["online", "offline", "disabled"],
        help="mode of wandb: online, offline, disabled",
        default="online",
    )
    parser.add_argument(
        "-train_dir",
        "--data_dir_train",
        type=str,
        help="directory of the train data",
        nargs="+",
        default="data_t(0, 0)_r(5, 15)_tennis_pNone_gNone",
    )
    parser.add_argument(
        "-l", "--loss", type=str, choices=["L1", "L2"], help="Loss type", default="L2"
    )
    parser.add_argument("--data_type", type=str, help="Type of data", default="pos")
    parser.add_argument(
        "-i", "--iterations", type=int, help="Number of iterations", default=1
    )
    parser.add_argument(
        "-extra_input",
        type=str,
        choices=[
            "inertia_body",
            "size",
            "size_squared",
            "size_mass",
            "size_squared_mass",
        ],
    )
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument(
        "--learning_rate", "-lr", type=float, default=0.0001, help="Batch size"
    )

    args = parser.parse_args()

    data_dir_train = "data/" + " ".join(args.data_dir_train)
    # data_dirs_test = args.data_dir_test]

    # data_dirs_test = os.listdir("data")
    # if ".DS_Store" in data_dirs_test:
    #     data_dirs_test.remove(".DS_Store")

    data_dirs_test = [
        " ".join(args.data_dir_train),
        "data_tennis_pNone_gNone_tennisEffect",
    ]
    print(f"Testing on datasets: {data_dirs_test}")

    # if args.data_dir_test == "":
    #     data_dirs_test = [data_dir_train]
    # else:
    #     data_dirs_test = "data/" + args.data_dir_test
    losses = [nn.MSELoss, nn.L1Loss]

    if not os.path.exists(data_dir_train):
        raise IndexError(f"No directory for the train data {data_dir_train}")

    ndata_dict = {
        "pos": 24,
        "eucl_motion": 12,
        "quat": 7,
        "log_quat": 7,
        "dual_quat": 8,
        "pos_diff": 24,
        "pos_diff_start": 24,
        "pos_norm": 24,
        "log_dualQ": 6,
    }
    n_extra_input = {
        "inertia_body": 3,
        "size": 3,
        "size_squared": 3,
        "size_mass": 4,
        "size_squared_mass": 4,
    }
    if args.extra_input:
        extra_input_n = n_extra_input[args.extra_input]
    else:
        extra_input_n = 0

    for i in range(args.iterations):
        print(f"----- ITERATION {i+1}/{args.iterations} ------")
        # Divide the train en test dataset
        n_sims_train = len(os.listdir(data_dir_train))
        n_sims_train = 4000
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
            learning_rate=args.learning_rate,
            epochs=20,
            batch_size=args.batch_size,
            loss_type=args.loss,
            loss_reduction_type="mean",
            optimizer="Adam",
            data_type=args.data_type,
            architecture="fcnn",
            train_sims=list(train_sims),
            test_sims=list(test_sims),
            n_frames=20,
            n_sims=n_sims_train,
            hidden_sizes=[128, 256],
            activation_func=["ReLU", "ReLU"],
            dropout=[0.0, 0.0],
            batch_norm=[False, False, False],
            lam=0.01,
            data_dir_train=data_dir_train,
            data_dirs_test=data_dirs_test,
            iter=i,
            # inertia_input=args.inertia_input,
            str_extra_input=args.extra_input,
            extra_input_n=extra_input_n,
        )

        loss_dict = {"L1": nn.L1Loss, "L2": nn.MSELoss}

        optimizer_dict = {"Adam": torch.optim.Adam}

        start_time = time.time()
        model = model_pipeline(
            config, ndata_dict, loss_dict, optimizer_dict, args.mode_wandb, losses
        )
        print(f"It took {time.time() - start_time} seconds to train & eval the model.")
        model_dict = {
            "config": config,
            "data_dict": ndata_dict,
            "model": model.state_dict(),
        }
        # Save model
        if not os.path.exists("models"):
            os.mkdir("models")
        torch.save(
            model_dict,
            f"models/fcnn/{config['data_type']}_{config['architecture']}_'{args.data_dir_train}'.pickle",
        )
