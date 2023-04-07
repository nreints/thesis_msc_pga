import torch
import torch.nn as nn
import torch.utils.data as data
from convert import *
import pickle
import random
import wandb
import time
import os
from general_functions import (
    model_pipeline,
    parse_args,
    get_data_dirs,
    divide_train_test_sims,
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class LSTM(nn.Module):
    def __init__(self, in_size, config):
        super().__init__()
        # Initialize the modules we need to build the network
        self.n_layers = config["n_layers"]
        self.hidden_size = config["hidden_size"]
        self.in_size = in_size
        if config["extra_input_n"] != 0:
            self.pre_hidden_lin_layer = nn.Sequential(
                nn.Linear(
                    config["extra_input_n"],
                    self.n_layers * self.hidden_size,
                )
            )
        self.lstm = nn.LSTM(
            in_size,
            self.hidden_size,
            batch_first=True,
            dropout=config["dropout"],
            num_layers=config["n_layers"],
        )
        self.post_lin_layers = nn.Sequential(nn.Linear(self.hidden_size, in_size))

    def forward(self, x, hidden_cell=None):
        # Perform the calculation of the model to determine the prediction

        batch_size, _, _ = x.shape
        if hidden_cell == None:
            hidden = None
            cell = None
        else:
            hidden, cell = hidden_cell
        if hidden == None:
            hidden_state = torch.zeros(
                self.n_layers, batch_size, self.hidden_size, device=device
            )
        else:
            # Map from inertia to hidden state
            hidden_state = self.pre_hidden_lin_layer(hidden).reshape(
                self.n_layers, batch_size, self.hidden_size
            )
        if cell == None:
            cell_state = torch.zeros(
                self.n_layers, batch_size, self.hidden_size, device=device
            )
        else:
            cell_state = cell
        out, h = self.lstm(x, (hidden_state, cell_state))
        return self.post_lin_layers(out), h


class MyDataset(data.Dataset):
    def __init__(self, sims, n_frames, n_data, data_type, dir, extra_input):
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
        count = 0
        for i in self.sims:
            with open(f"{self.dir}/sim_{i}.pickle", "rb") as f:
                data_all = pickle.load(f)["data"]
                data = torch.FloatTensor(data_all[self.data_type][:500])
                if count == 0:
                    data_per_sim = len(data) - (self.n_frames_perentry + 1)
                    len_data = len(self.sims) * data_per_sim
                    self.target = torch.zeros(
                        (len_data, self.n_frames_perentry, self.n_datap_perframe)
                    )
                    self.target_pos = torch.zeros(
                        (len_data, self.n_frames_perentry, 24)
                    )
                    self.start_pos = torch.zeros((len_data, 24))
                    self.data = torch.zeros(
                        len_data, self.n_frames_perentry, self.n_datap_perframe
                    )
                    self.extra_input_data = torch.zeros((len_data, 3))
                    self.xpos_start = torch.zeros((len_data, 3))
                for frame in range(len(data) - (self.n_frames_perentry + 1)):
                    self.start_pos[count] = torch.FloatTensor(
                        data_all["pos"][0].flatten()
                    )
                    self.xpos_start[count] = torch.FloatTensor(
                        data_all["xpos_start"].flatten()
                    )
                    train_end = frame + self.n_frames_perentry
                    self.data[count] = data[frame:train_end].reshape(
                        -1, self.n_datap_perframe
                    )
                    self.target[count] = data[frame + 1 : train_end + 1].reshape(
                        -1, self.n_datap_perframe
                    )

                    if self.extra_input[1] != 0:
                        extra_input_values = torch.FloatTensor(
                            data_all[self.extra_input[0]]
                        )
                        self.extra_input_data[count] = extra_input_values
                    self.target_pos[count] = torch.FloatTensor(
                        data_all["pos"][frame + 1 : train_end + 1]
                    ).flatten(start_dim=1)
                    count += 1

        self.normalize_extra_input = torch.mean(
            torch.norm(self.extra_input_data, dim=1)
        )

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        # Return the idx-th data point of the dataset
        data_point = self.data[idx]
        data_target = self.target[idx]
        data_target_pos = self.target_pos[idx]
        data_start = self.start_pos[idx]
        extra_input_data = self.extra_input_data[idx]
        start_xpos = self.xpos_start[idx]
        return (
            data_point,
            data_target,
            data_target_pos,
            data_start,
            extra_input_data,
            start_xpos,
        )


def train_log(loss, epoch):
    wandb.log({"Epoch": epoch, "Train loss": loss}, step=epoch)
    # print(f"Loss after " + f" examples: {loss:.3f}")


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
    print("-- Started Training --")
    # Set model to train mode
    model.train()
    wandb.watch(model, loss_module, log="all", log_freq=10)

    # Training loop
    for epoch in range(num_epochs):
        loss_epoch = 0
        epoch_time = time.time()

        for (
            data_inputs,
            data_labels,
            pos_target,
            start_pos,
            extra_input_data,
            xpos_start,
        ) in data_loader:
            data_inputs = data_inputs.to(device)  # Shape: [batch, frames, n_data]
            data_labels = data_labels.to(device)  # Shape: [batch, frames, n_data]
            pos_target = pos_target.to(device)  # Shape: [batch, frames, n_data]
            start_pos = start_pos.to(device)  # Shape: [batch, n_data]
            extra_input_data = extra_input_data.to(device)  # Shape: [batch, 3]

            if config["str_extra_input"] == "inertia_body":
                extra_input_data /= normalization

            if config.extra_input_n != 0:
                preds, _ = model(
                    data_inputs, (extra_input_data, None)
                )  # Shape: [batch, frames, n_data]
            else:
                preds, _ = model(data_inputs)  # Shape: [batch, frames, n_data]

            if config.data_type[-3:] != "ori":
                alt_preds = convert(
                    preds,
                    start_pos,
                    data_loader.dataset.data_type,
                    xpos_start,
                )

            else:
                alt_preds = convert(
                    preds,
                    start_pos,
                    data_loader.dataset.data_type,
                )

            assert not torch.any(
                torch.isnan(alt_preds)
            ), f"Encountered NaN in alt_preds."

            assert not torch.any(
                torch.isnan(pos_target)
            ), f"Encountered NaN in alt_preds."

            loss = loss_module(alt_preds, pos_target)

            optimizer.zero_grad()

            # Perform backpropagation
            loss.backward()

            optimizer.step()

            loss_epoch += loss

        print(f"Epoch {epoch}")
        train_log(loss_epoch / len(data_loader), epoch)
        print(
            f"\t Logging train Loss: {round(loss_epoch.item() / len(data_loader), 10)} ({loss_module}: {config.data_dir_train[5:]})"
        )

        convert_loss = eval_model(
            model, test_loaders, config, epoch, losses, normalization
        )
        model.train()
        print(f"     --> Epoch time; {time.time() - epoch_time}")


def eval_model(model, data_loaders, config, current_epoch, losses, normalization):
    model.eval()  # Set model to eval mode

    with torch.no_grad():  # Deactivate gradients for the following code
        for i, data_loader in enumerate(data_loaders):
            for loss_module in losses:
                loss_module = loss_module(reduction=config.loss_reduction_type)
                total_loss = 0
                total_convert_loss = 0
                for (
                    data_inputs,
                    data_labels,
                    data_labels_pos,
                    start_pos,
                    extra_input_data,
                    xpos_start,
                ) in data_loader:

                    # Determine prediction of model on dev set
                    data_inputs = data_inputs.to(device)
                    data_labels = data_labels.to(device)
                    extra_input_data = extra_input_data.to(device)
                    if config["str_extra_input"] == "inertia_body":
                        extra_input_data /= normalization

                    if config.extra_input_n != 0:
                        preds, _ = model(
                            data_inputs, (extra_input_data, None)
                        )  # Shape: [batch, frames, n_data]
                    else:
                        preds, _ = model(data_inputs)  # Shape: [batch, frames, n_data]
                    preds = preds.squeeze(dim=1)

                    if config.data_type[-3:] != "ori":
                        alt_preds = convert(
                            preds.detach().cpu(),
                            start_pos,
                            data_loader.dataset.data_type,
                            xpos_start,
                        )

                    else:
                        alt_preds = convert(
                            preds,
                            start_pos,
                            data_loader.dataset.data_type,
                        )

                    total_loss += loss_module(preds, data_labels)
                    total_convert_loss += loss_module(alt_preds, data_labels_pos)

                print(
                    f"\t Logging test loss: {total_convert_loss / len(data_loader)} ({loss_module}: {config.data_dirs_test[i][5:]})"
                )
                if config.data_dirs_test[i] == config.data_dir_train[5:]:
                    extra_wandb_string = ""
                else:
                    extra_wandb_string = " " + config.data_dirs_test[i][5:]

                wandb.log(
                    {
                        f"Test loss{extra_wandb_string}": total_convert_loss
                        / len(data_loader)
                    },
                    step=current_epoch,
                )

    return total_convert_loss.item() / len(data_loader)


if __name__ == "__main__":
    args = parse_args()

    data_train_dir, data_dirs_test = get_data_dirs(args.data_dir_train)
    data_dir_train = "data/" + data_train_dir
    if not os.path.exists(data_dir_train):
        raise IndexError("No directory for the train data {args.data_dir_train}")

    ndata_dict = {
        "pos": 24,
        "rot_mat": 12,
        "quat": 7,
        "log_quat": 7,
        "dual_quat": 8,
        "pos_diff": 24,
        "pos_diff_start": 24,
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

    losses = [nn.MSELoss]

    for i in range(args.iterations):
        print(f"----- ITERATION {i+1}/{args.iterations} ------")
        # Divide the train en test dataset
        n_sims_train_total, train_sims, test_sims = divide_train_test_sims(
            data_dir_train, data_dirs_test
        )

        config = dict(
            learning_rate=args.learning_rate,
            epochs=10,
            batch_size=args.batch_size,
            dropout=0.0,
            loss_type=args.loss,
            loss_reduction_type="mean",
            optimizer="Adam",
            data_type=args.data_type,
            architecture="lstm",
            train_sims=list(train_sims),
            test_sims=list(test_sims),
            n_frames=20,
            n_sims=n_sims_train_total,
            n_layers=1,
            hidden_size=96,
            data_dir_train=data_train_dir,
            data_dirs_test=data_dirs_test,
            iter=i,
            str_extra_input=args.extra_input,
            extra_input_n=extra_input_n,
        )

        start_time = time.time()
        model = model_pipeline(
            config,
            ndata_dict,
            args.mode_wandb,
            losses,
            train_model,
            device,
            MyDataset,
            LSTM,
        )
        print("It took ", time.time() - start_time, " seconds.")
