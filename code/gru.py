import os
import time

import torch
import torch.nn as nn

import wandb
from convert import *
from dataset import RecurrentDataset
from utils import *

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class GRU(nn.Module):
    def __init__(
        self,
        input_shape,
        config,
        num_outputs=0,
    ):
        """
        Construct a QuaterNet neural network.
        Arguments:
         -- num_joints: number of skeleton joints.
         -- num_outputs: extra inputs/outputs (e.g. translations), in addition to joint rotations.
         -- model_velocities: add a quaternion multiplication block on the RNN output to force
                              the network to model velocities instead of absolute rotations.
        """
        super().__init__()

        self.num_outputs = num_outputs
        self.n_data = input_shape
        self.n_layers = config["n_layers"]

        self.h_size = config["hidden_size"]
        self.rnn = nn.GRU(
            input_size=input_shape,
            hidden_size=self.h_size,
            num_layers=self.n_layers,
            batch_first=True,
        )
        if config["extra_input_n"] != 0:
            self.pre_hidden_lin_layer = nn.Sequential(
                nn.Linear(
                    config["extra_input_n"],
                    self.n_layers * self.h_size,
                ),
                nn.ReLU(),
                nn.Linear(
                    self.n_layers * self.h_size,
                    self.n_layers * self.h_size,
                ),
                nn.ReLU(),
            )
        self.h0 = nn.Parameter(
            torch.zeros(self.n_layers, 1, self.h_size).normal_(std=0.01),
            requires_grad=True,
        )

        self.fc = nn.Linear(self.h_size, input_shape)

    def forward(self, x, h=None, return_all=True):
        """
        Run a forward pass of this model.
        Arguments:
         -- x: input tensor of shape (N, L, !!!!!!J*4 + O + C), where N is the batch size, L is the sequence length,
               J is the number of joints, O is the number of outputs, and C is the number of controls.
               Features must be provided in the order J, O, C.
         -- h: hidden state. If None, it defaults to the learned initial state.
         -- return_prenorm: if True, return the quaternions prior to normalization.
         -- return_all: if True, return all L frames, otherwise return only the last frame. If only the latter
                        is wanted (e.g. when conditioning the model with an initialization sequence), this
                        argument should be left to False as it avoids unnecessary computation.
        """
        assert len(x.shape) == 3

        x_orig = x

        if h is None:
            h = self.h0.expand(-1, x.shape[0], -1).contiguous()
        else:
            h = self.pre_hidden_lin_layer(h).reshape(
                self.n_layers, x.shape[0], self.h_size
            )
        x, h = self.rnn(x, h)

        if return_all:
            x = self.fc(x)
        else:
            x = self.fc(x[:, -1:])
            x_orig = x_orig[:, -1:]

        pre_normalized = x[:, :, : self.n_data].contiguous()
        return x, h, torch.cat((pre_normalized, x[:, :, self.n_data :]), dim=2)


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
            data_inputs = data_inputs.to(DEVICE)  # Shape: [batch, frames, n_data]
            data_labels = data_labels.to(DEVICE)  # Shape: [batch, frames, n_data]
            pos_target = pos_target.to(DEVICE)  # Shape: [batch, frames, n_data]
            start_pos = start_pos.to(DEVICE)  # Shape: [batch, n_data]
            extra_input_data = extra_input_data.to(DEVICE)  # Shape: [batch, 3]
            xpos_start = xpos_start.to(DEVICE)

            if config["str_extra_input"]:
                extra_input_data /= normalization

            if config.extra_input_n != 0:
                _, _, preds = model(
                    data_inputs, extra_input_data
                )  # Shape: [batch, frames, n_data]
            else:
                _, _, preds = model(data_inputs)  # Shape: [batch, frames, n_data]

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
        print(f"\t\t--> Epoch time; {time.time() - epoch_time}")


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
                    data_inputs = data_inputs.to(DEVICE)
                    data_labels = data_labels.to(DEVICE)
                    extra_input_data = extra_input_data.to(DEVICE)

                    if config["str_extra_input"]:
                        extra_input_data /= normalization

                    if config.extra_input_n != 0:
                        _, _, preds = model(
                            data_inputs, extra_input_data
                        )  # Shape: [batch, frames, n_data]
                    else:
                        _, _, preds = model(
                            data_inputs
                        )  # Shape: [batch, frames, n_data]

                    # Convert predictions to xyz-data
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
        raise IndexError(f"No directory for the train data {data_dir_train}")

    extra_input_n = extra_input(args.inertia_body)
    reference = get_reference(args.data_type)

    print(
        f"Focussing on identity: {args.focus_identity}\nUsing extra input: {args.extra_input}\nUsing {reference} as reference point."
    )

    losses = [nn.MSELoss]

    for i in range(args.iterations):
        print(f"----- ITERATION {i+1}/{args.iterations} ------")
        # Divide the train en test dataset
        n_sims_train_total, train_sims, test_sims = divide_train_test_sims(
            data_dir_train, data_dirs_test, "train_test_ids_2400"
        )

        config = dict(
            learning_rate=args.learning_rate,
            epochs=10,
            batch_size=args.batch_size,
            loss_type=args.loss,
            loss_reduction_type="mean",
            optimizer="Adam",
            data_type=args.data_type,
            reference=reference,
            architecture="gru",
            train_sims=train_sims,
            test_sims=test_sims,
            n_frames=args.input_frames,
            n_sims=n_sims_train_total,
            n_layers=1,
            hidden_size=96,
            data_dir_train=data_train_dir,
            data_dirs_test=data_dirs_test,
            iter=i,
            str_extra_input=args.inertia_body,
            extra_input_n=extra_input_n,
            focus_identity=args.focus_identity,
            fix_determinant=True,
        )

        start_time = time.time()
        model = model_pipeline(
            config,
            args.mode_wandb,
            losses,
            train_model,
            DEVICE,
            RecurrentDataset,
            GRU,
            args.wandb_name,
        )
        print("It took ", time.time() - start_time, " seconds.")
