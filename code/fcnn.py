import torch
import torch.nn as nn
import pickle
import random
import torch.utils.data as data
from convert import *
import wandb
import time
import os
from general_functions import (
    model_pipeline,
    parse_args,
    get_data_dirs,
    divide_train_test_sims,
    nr_extra_input,
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class fcnn(nn.Module):
    def __init__(self, n_data, config):
        super().__init__()

        # Add first layers
        self.layers = [
            nn.Linear(
                config["n_frames"] * n_data + config["extra_input_n"],
                config["hidden_sizes"][0],
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
    ):
        """
        Inputs:
            - sims; simulation IDs to use in this dataset
            - n_frames; number of input frames
            - n_data; number of datapoints given the data_type
            - data_type; type of the data
            - dir; directory where the data is stored
            - extra_input; tuple
                - extra_input[0]; type of extra input
                - extra_input[1]; number of extra input values
        """
        super().__init__()
        self.n_frames_perentry = n_frames
        self.n_datap_perframe = n_data
        self.sims = sims
        self.data_type = data_type
        self.dir = dir
        self.extra_input = extra_input
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
                    self.target = torch.zeros((len_data, self.n_datap_perframe))
                    self.target_pos = torch.zeros((len_data, 24))
                    self.start_pos = torch.zeros_like(self.target_pos)
                    self.xpos_start = torch.zeros((len_data, 3))
                for frame in range(data_per_sim):
                    # Always save the start position for converting
                    self.start_pos[count] = torch.FloatTensor(pos_data[0].flatten())
                    self.xpos_start[count] = torch.FloatTensor(
                        data_all["xpos_start"].flatten()
                    )
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

        # self.mean = torch.mean(self.data)
        # self.std = torch.std(self.data)
        # self.normalized_data = (self.data - self.mean) / self.std

        self.normalize_extra_input = torch.mean(
            torch.norm(self.data[:, -self.extra_input[1] :], dim=1)
        )
        assert (
            self.normalize_extra_input != 0
        ), f"The normalization of the extra input is zero. This leads to zero-division."
        # print(self.xpos_start.shape)
        # print("mean of norm extra_input", self.normalize_extra_input.item())
        # self.data[:, -self.extra_input[1] :] = self.extra_input_data
        print(f"The dataloader took {time.time() - start_time} seconds.")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        # Return the idx-th data point of the dataset
        data_point = self.data[idx]
        data_target = self.target[idx]
        data_start = self.start_pos[idx]
        data_pos_target = self.target_pos[idx]
        start_xpos = self.xpos_start[idx]
        return data_point, data_target, data_start, data_pos_target, start_xpos


def train_log(loss, epoch, config):
    """
    Log the train loss to Weights and Biases
    """
    wandb.log({f"Train loss": loss}, step=epoch)


def train_model(
    model,
    optimizer,
    data_loader,
    test_loaders,
    loss_module,
    num_epochs,
    config,
    losses,
    normalization,
):
    print("--- Started Training ---")
    # Set model to train mode
    model.train()
    wandb.watch(model, loss_module, log="all", log_freq=10)

    # Training loop
    for epoch in range(num_epochs):
        epoch_time = time.time()
        loss_epoch = 0
        for (
            data_inputs,
            data_labels,
            start_pos,
            pos_target,
            xpos_start,
        ) in data_loader:

            # Set data to current device
            data_inputs = data_inputs.to(
                device
            )  # Shape: [batch, frames x n_data + config["extra_input_n"]]
            assert not torch.any(
                torch.isnan(data_inputs)
            ), f"Encountered NaN in the data inputs."
            data_labels = data_labels.to(device)  # Shape: [batch, n_data]
            pos_target = pos_target.to(device)  # Shape: [batch, n_data]
            start_pos = start_pos.to(device)  # Shape: [batch, n_data]
            xpos_start = xpos_start.to(device)
            if config["str_extra_input"] == "inertia_body":
                data_inputs[:, -config["extra_input_n"] :] = (
                    data_inputs[:, -config["extra_input_n"] :] / normalization
                )

            # Get predictions
            preds = model(data_inputs)  # Shape: [batch, n_data]
            # preds = model(data_norm)
            # preds = preds * data_set_train.std + data_set_train.mean

            # Convert predictions to xyz-data
            if config.data_type[-3:] != "ori":
                alt_preds = convert(
                    preds,
                    start_pos,
                    data_loader.dataset.data_type,
                    xpos_start,
                )
            # print(alt_preds)

            else:
                alt_preds = convert(
                    preds,
                    start_pos,
                    data_loader.dataset.data_type,
                )
                # print("alt_preds", alt_preds[0][0])
            # print("alt_preds:", alt_preds[0])
            # print("pos_targ", pos_target[0])
            if torch.any(torch.isnan(preds)):
                print("5")
                exit()

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

            loss_epoch += position_loss

            # Perform backpropagation
            optimizer.zero_grad()
            # loss = torch.tensor(loss, requires_grad=True)
            loss.backward()

            optimizer.step()
        # if epoch == 5 or epoch == 10:
        #     print(data_inputs[0].reshape(20, 12)[-2:])
        #     print(data_labels[0])
        #     print(pos_target[0].shape)
        #     print(preds[0])
        #     exit()

        print(f"Epoch {epoch}")
        # Log to W&B
        train_log(loss_epoch / len(data_loader), epoch, config)
        print(
            f"\t Logging train Loss: {round(loss_epoch.item() / len(data_loader), 10)} ({loss_module}: {config.data_dir_train[5:]})"
        )

        # Evaluate model
        true_loss, convert_loss, total_convert_loss = eval_model(
            model, test_loaders, loss_module, config, epoch, losses, normalization
        )

        # Set model to train mode
        model.train()
        print(f"     --> Epoch time; {time.time() - epoch_time}")


def eval_model(
    model, data_loaders, loss_module, config, current_epoch, losses, normalization
):

    model.eval()  # Set model to eval mode

    with torch.no_grad():  # Deactivate gradients for the following code
        for i, data_loader in enumerate(data_loaders):
            for loss_module in losses:
                loss_module = loss_module(reduction=config.loss_reduction_type)
                total_loss = 0
                total_convert_loss = 0
                wandb_total_convert_loss = 0
                for (
                    data_inputs,
                    data_labels,
                    start_pos,
                    pos_target,
                    xpos_start,
                ) in data_loader:
                    if config["str_extra_input"] == "inertia_body":
                        data_inputs[:, -config["extra_input_n"] :] = (
                            data_inputs[:, -config["extra_input_n"] :] / normalization
                        )
                    # Set data to current device
                    data_inputs = data_inputs.to(device)
                    # data_norm = (data_inputs - data_set_train.mean) / data_set_train.std
                    data_labels = data_labels.to(device)
                    xpos_start = xpos_start.to(device)
                    # Get predictions
                    preds = model(data_inputs)
                    # preds = model(data_norm)
                    preds = preds.squeeze(dim=1)
                    # preds = preds * data_set_train.std + data_set_train.mean
                    # print(data_labels[0][-3:])

                    # Convert predictions to xyz-data
                    if config.data_type[-3:] != "ori":
                        alt_preds = convert(
                            preds.detach().cpu(),
                            start_pos,
                            data_loader.dataset.data_type,
                            xpos_start,
                        )
                    else:
                        alt_preds = convert(
                            preds.detach().cpu(),
                            start_pos,
                            data_loader.dataset.data_type,
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
                if config.data_dirs_test[i] == config.data_dir_train[5:]:
                    extra_wandb_string = ""
                else:
                    extra_wandb_string = " " + config.data_dirs_test[i][5:]

                wandb.log(
                    {
                        f"Test loss{extra_wandb_string}": wandb_total_convert_loss
                        / len(data_loader)
                    },
                    step=current_epoch,
                )
    # Return the average loss
    return (
        total_loss.item() / len(data_loader),
        wandb_total_convert_loss.item() / len(data_loader),
        total_convert_loss.item() / len(data_loader),
    )


if __name__ == "__main__":
    args = parse_args()

    data_train_dir, data_dirs_test = get_data_dirs(args.data_dir_train)
    data_dir_train = "data/" + data_train_dir

    if not os.path.exists(data_dir_train):
        raise IndexError(f"No directory for the train data {data_dir_train}")

    extra_input_n = nr_extra_input(args.extra_input)

    losses = [nn.MSELoss]

    for i in range(args.iterations):
        print(f"----- ITERATION {i+1}/{args.iterations} ------")
        # Divide the train en test dataset
        n_sims_train_total, train_sims, test_sims = divide_train_test_sims(
            data_dir_train, data_dirs_test
        )
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
            n_sims=n_sims_train_total,
            hidden_sizes=[128, 256],
            activation_func=["Tanh", "ReLU"],
            dropout=[0, 0],
            batch_norm=[False, False, False],
            lam=0.01,
            data_dir_train=data_train_dir,
            data_dirs_test=data_dirs_test,
            iter=i,
            str_extra_input=args.extra_input,
            extra_input_n=extra_input_n,
        )

        start_time = time.time()
        model_pipeline(
            config,
            args.mode_wandb,
            losses,
            train_model,
            device,
            MyDataset,
            fcnn,
        )
        print(f"It took {time.time() - start_time} seconds to train & eval the model.")
