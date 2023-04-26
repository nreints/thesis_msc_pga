import os
import pickle
import random
import time

import torch
import torch.nn as nn
import torch.utils.data as data

import wandb
from convert import *
from dataset import NonRecurrentDataset
from utils import *

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
            alt_preds = convert(
                preds,
                start_pos,
                config.data_type,
                xpos_start,
            )
            # print("alt_preds", alt_preds[0][0])
            # print("alt_preds:", alt_preds[0])
            # print("pos_targ", pos_target[0])
            if torch.any(torch.isnan(preds)):
                print("FCNN, train, NaN")
                exit()

            # Determine norm penalty for quaternion data
            # if config["data_type"] == "quat" or config["data_type"] == "dual_quat":
            #     norm_penalty = (
            #         config["lam"]
            #         * (1 - torch.mean(torch.norm(preds[:, :4], dim=-1))) ** 2
            #     )
            # else:
            #     norm_penalty = 0

            position_loss = loss_module(alt_preds, pos_target)
            # Calculate the total loss
            loss = position_loss  # + norm_penalty

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

        print(f"Epoch {epoch}/{num_epochs-1}")
        # Log to W&B
        train_log(
            loss_epoch / len(data_loader), epoch, loss_module, config.data_dir_train[5:]
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

                    # Convert predictions to xyz-data
                    alt_preds = convert(
                        preds.detach().cpu(),
                        start_pos,
                        config.data_type,
                        xpos_start,
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
                eval_log(
                    config.data_dirs_test[i],
                    config.data_dir_train,
                    total_convert_loss / len(data_loader),
                    current_epoch,
                    loss_module,
                )

    # Return the average loss
    return (
        total_loss.item() / len(data_loader),
        wandb_total_convert_loss.item() / len(data_loader),
        total_convert_loss.item() / len(data_loader),
    )


if __name__ == "__main__":
    args = parse_args()
    print(args.data_dirs_test)

    data_train_dir, data_dirs_test = get_data_dirs(
        args.data_dir_train, args.data_dirs_test
    )
    data_dir_train = "data/" + data_train_dir

    if not os.path.exists(data_dir_train):
        raise IndexError(f"No directory for the train data {data_dir_train}")

    extra_input_n = nr_extra_input(args.extra_input)
    reference = get_reference(args.data_type)

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
            epochs=10,
            batch_size=args.batch_size,
            loss_type=args.loss,
            loss_reduction_type="mean",
            optimizer="Adam",
            data_type=args.data_type,
            architecture="fcnn",
            train_sims=train_sims,
            test_sims=test_sims,
            n_frames=args.input_frames,
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
            wrt=reference,
        )

        start_time = time.time()
        model_pipeline(
            config,
            args.mode_wandb,
            losses,
            train_model,
            device,
            NonRecurrentDataset,
            fcnn,
        )
        print(f"It took {time.time() - start_time} seconds to train & eval the model.")
