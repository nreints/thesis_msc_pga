import os
import time

import torch
import torch.nn as nn

import wandb
from convert import *
from dataset import RecurrentDataset
from utils import *

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

            alt_preds = convert(
                preds, start_pos, config.data_type, xpos_start, config["focus_identity"]
            )

            assert not torch.any(
                torch.isnan(alt_preds)
            ), f"Encountered NaN in alt_preds."

            assert not torch.any(
                torch.isnan(pos_target)
            ), f"Encountered NaN in pos_target."

            loss = loss_module(alt_preds, pos_target)

            optimizer.zero_grad()

            # Perform backpropagation
            loss.backward()

            optimizer.step()

            loss_epoch += loss

        print(f"Epoch {epoch}/{num_epochs-1}")
        train_log(
            loss_epoch / len(data_loader), epoch, loss_module, config.data_dir_train[5:]
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

                    alt_preds = convert(
                        preds.detach().cpu(),
                        start_pos,
                        config.data_type,
                        xpos_start,
                        config["focus_identity"],
                    )

                    total_loss += loss_module(preds, data_labels)
                    total_convert_loss += loss_module(alt_preds, data_labels_pos)

                eval_log(
                    config.data_dirs_test[i],
                    config.data_dir_train,
                    total_convert_loss / len(data_loader),
                    current_epoch,
                    loss_module,
                )

    return total_convert_loss.item() / len(data_loader)


if __name__ == "__main__":
    args = parse_args()

    data_train_dir, data_dirs_test = get_data_dirs(
        args.data_dir_train, args.data_dirs_test
    )
    data_dir_train = "data/" + data_train_dir

    if not os.path.exists(data_dir_train):
        raise IndexError(f"No directory for the train data {args.data_dir_train}")

    extra_input_n = nr_extra_input(args.extra_input)
    reference = get_reference(args.data_type)

    print(
        f"Focussing on identity: {args.focus_identity}\nUsing extra input: {args.extra_input}\nUsing {reference} as reference point."
    )

    losses = [nn.MSELoss]

    for i in range(args.iterations):
        print(f"----- ITERATION {i+1}/{args.iterations} ------")
        # Divide the train en test dataset
        n_sims_train_total, train_sims, test_sims = divide_train_test_sims(
            data_dir_train, data_dirs_test, "train_test_ids_1000", i
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
            reference=reference,
            architecture="lstm",
            train_sims=train_sims,
            test_sims=test_sims,
            n_frames=args.input_frames,
            n_sims=n_sims_train_total,
            n_layers=1,
            hidden_size=96,
            data_dir_train=data_train_dir,
            data_dirs_test=data_dirs_test,
            iter=i,
            str_extra_input=args.extra_input,
            extra_input_n=extra_input_n,
            focus_identity=args.focus_identity,
        )

        start_time = time.time()
        model = model_pipeline(
            config,
            args.mode_wandb,
            losses,
            train_model,
            device,
            RecurrentDataset,
            LSTM,
            args.wandb_name,
        )
        print("It took ", time.time() - start_time, " seconds.")
