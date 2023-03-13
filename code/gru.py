import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as data
from convert import *
import pickle
import random
import wandb
import time
import os
import argparse

# from quaternion import qmul
# import torch.nn.functional as F

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class GRU(nn.Module):
    def __init__(
        self, config, input_shape, num_outputs=0, num_controls=0, model_velocities=False
    ):
        """
        Construct a QuaterNet neural network.
        Arguments:
         -- num_joints: number of skeleton joints.
         -- num_outputs: extra inputs/outputs (e.g. translations), in addition to joint rotations.
         -- num_controls: extra input-only features.
         -- model_velocities: add a quaternion multiplication block on the RNN output to force
                              the network to model velocities instead of absolute rotations.
        """
        super().__init__()

        # self.num_joints = num_joints
        self.num_outputs = num_outputs
        self.num_controls = num_controls
        self.n_data = input_shape
        self.n_layers = config["n_layers"]

        # if num_controls > 0:
        #     fc1_size = 30
        #     fc2_size = 30
        #     self.fc1 = nn.Linear(num_controls, fc1_size)
        #     self.fc2 = nn.Linear(fc1_size, fc2_size)
        #     self.relu = nn.LeakyReLU(0.05, inplace=True)
        # else:
        #     fc2_size = 0

        self.h_size = config["hidden_size"]
        self.rnn = nn.GRU(
            input_size=input_shape,
            hidden_size=self.h_size,
            num_layers=self.n_layers,
            batch_first=True,
        )
        self.pre_hidden_lin_layer = nn.Sequential(
            nn.Linear(
                3,
                self.n_layers * self.h_size,
            )
        )
        self.h0 = nn.Parameter(
            torch.zeros(self.n_layers, 1, self.h_size).normal_(std=0.01),
            requires_grad=True,
        )

        self.fc = nn.Linear(self.h_size, input_shape)

    def forward(self, x, h=None, return_prenorm=True, return_all=True):
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
        # assert x.shape[-1] == self.num_joints*4 + self.num_outputs + self.num_controls

        x_orig = x
        # print(self.num_controls)
        # if self.num_controls > 0:
        #     controls = x[:, :, self.num_joints*4+self.num_outputs:]
        #     controls = self.relu(self.fc1(controls))
        #     controls = self.relu(self.fc2(controls))
        #     x = torch.cat((x[:, :, :self.num_joints*4+self.num_outputs], controls), dim=2)
        # print(x.shape)

        if h is None:
            h = self.h0.expand(-1, x.shape[0], -1).contiguous()
        else:
            h = self.pre_hidden_lin_layer(h).reshape(
                self.n_layers, x.shape[0], self.h_size
            )
        x, h = self.rnn(x, h)
        # print(x.shape, "H")
        if return_all:
            x = self.fc(x)
        else:
            x = self.fc(x[:, -1:])
            x_orig = x_orig[:, -1:]

        # print(x.shape, "l")

        pre_normalized = x[:, :, : self.n_data].contiguous()
        # normalized = pre_normalized.view(-1, 4)
        # if self.model_velocities:
        #     normalized = qmul(normalized, x_orig[:, :, :self.num_joints*4].contiguous().view(-1, 4))
        # normalized = F.normalize(normalized, dim=1).view(pre_normalized.shape)

        # if self.num_outputs > 0:
        #     x = torch.cat((normalized, x[:, :, self.num_joints*4:]), dim=2)
        # else:
        #     x = normalized

        # if return_prenorm:
        return x, h, torch.cat((pre_normalized, x[:, :, self.n_data :]), dim=2)
        # else:
        # return x, h


class MyDataset(data.Dataset):
    def __init__(self, sims, n_frames, n_data, data_type, dir, input_inertia=True):
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
        self.dir = dir
        self.input_inertia = input_inertia
        self.collect_data()

    def collect_data(self):
        # start_time = time.time()
        # self.data = []
        # self.target = []
        # self.target_pos = []
        # self.start_pos = []

        # for i in self.sims:
        #     with open(f"{self.dir}/sim_{i}.pickle", "rb") as f:
        #         data_all = pickle.load(f)["data"]
        #         data = data_all[self.data_type]
        #         for frame in range(len(data) - (self.n_frames_perentry + 1)):
        #             self.start_pos.append(data_all["pos"][0].flatten())
        #             train_end = frame + self.n_frames_perentry
        #             # [frames, n_data]
        #             self.data.append(
        #                 data[frame:train_end].reshape(-1, self.n_datap_perframe)
        #             )
        #             # [frames, n_data]
        #             self.target.append(
        #                 data[frame + 1 : train_end + 1].reshape(
        #                     -1, self.n_datap_perframe
        #                 )
        #             )
        #             # [frames, 8, 3]
        #             self.target_pos.append(data_all["pos"][frame + 1 : train_end + 1])

        # # Shape [(n_simsx(total_nr_frames-n_frames_perentry-1)), n_frames_perentry, n_data]
        # self.data = torch.FloatTensor(np.asarray(self.data))
        # self.target = torch.FloatTensor(np.asarray(self.target))
        # self.target_pos = torch.FloatTensor(np.asarray(self.target_pos)).flatten(
        #     start_dim=2
        # )
        # self.start_pos = torch.FloatTensor(np.asarray(self.start_pos))

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
                    self.inertia = torch.zeros((len_data, 3))
                for frame in range(len(data) - (self.n_frames_perentry + 1)):
                    self.start_pos[count] = torch.FloatTensor(
                        data_all["pos"][0].flatten()
                    )
                    train_end = frame + self.n_frames_perentry
                    self.data[count] = data[frame:train_end].reshape(
                        -1, self.n_datap_perframe
                    )
                    self.target[count] = data[frame + 1 : train_end + 1].reshape(
                        -1, self.n_datap_perframe
                    )

                    if self.input_inertia:
                        # TODO
                        # inertia = data_all["inertia"]
                        inertia = torch.tensor([1, 2, 3])
                        self.inertia[count] = inertia
                    self.target_pos[count] = torch.FloatTensor(
                        data_all["pos"][frame + 1 : train_end + 1]
                    ).flatten(start_dim=1)
                    count += 1
        # print(time.time() - start_time)
        # exit()

    def __len__(self):
        # Number of data point we have. Alternatively self.data.shape[0], or self.label.shape[0]
        return self.data.shape[0]

    def __getitem__(self, idx):
        # Return the idx-th data point of the dataset
        data_point = self.data[idx]
        data_target = self.target[idx]
        data_target_pos = self.target_pos[idx]
        data_start = self.start_pos[idx]
        inertia = self.inertia[idx]
        return data_point, data_target, data_target_pos, data_start, inertia


def train_log(loss, epoch):
    wandb.log({"Epoch": epoch, "Train loss": loss}, step=epoch)
    # print(f"Loss after " + f" examples: {loss:.3f}")


def train_model(
    model, optimizer, data_loader, test_loaders, loss_module, num_epochs, config, losses
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
            inertia_input,
        ) in data_loader:
            # start = time.time()

            data_inputs = data_inputs.to(device)  # Shape: [batch, frames, n_data]
            data_labels = data_labels.to(device)  # Shape: [batch, frames, n_data]
            pos_target = pos_target.to(device)  # Shape: [batch, frames, n_data]
            start_pos = start_pos.to(device)  # Shape: [batch, n_data]
            inertia_input = inertia_input.to(device)  # Shape: [batch, 3]

            if config.inertia_input:
                _, _, preds = model(
                    data_inputs, inertia_input
                )  # Shape: [batch, frames, n_data]
            else:
                _, _, preds = model(data_inputs)  # Shape: [batch, frames, n_data]

            alt_preds = convert(preds, start_pos, data_loader.dataset.data_type)

            loss = loss_module(alt_preds, pos_target)

            optimizer.zero_grad()

            # Perform backpropagation
            loss.backward()

            optimizer.step()

            loss_epoch += loss

        train_log(loss_epoch / len(data_loader), epoch)

        convert_loss = eval_model(model, test_loaders, config, epoch, losses)
        model.train()
        print(
            epoch,
            round(loss_epoch.item() / len(data_loader), 10),
            "\t",
            round(convert_loss, 10),
        )
        print("epoch_time; ", time.time() - epoch_time)


def eval_model(model, data_loaders, config, current_epoch, losses):
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
                    inertia_input,
                ) in data_loader:

                    # Determine prediction of model on dev set
                    data_inputs = data_inputs.to(device)
                    data_labels = data_labels.to(device)
                    inertia_input = inertia_input.to(device)

                    if config.inertia_input:
                        _, _, preds = model(
                            data_inputs, inertia_input
                        )  # Shape: [batch, frames, n_data]
                    else:
                        _, _, preds = model(
                            data_inputs
                        )  # Shape: [batch, frames, n_data]
                        preds = preds.squeeze(dim=1)

                    # if config['data_type'] == 'pos':
                    #     preds = preds.reshape((preds.shape[0], preds.shape[1], 8, 3))
                    alt_preds = convert(
                        preds.detach().cpu(), start_pos, data_loader.dataset.data_type
                    )

                    total_loss += loss_module(preds, data_labels)
                    total_convert_loss += loss_module(alt_preds, data_labels_pos)

                print(
                    f"\t Logging test loss: {config.data_dirs_test[i][5:]}, {str(loss_module)} => {round((total_convert_loss / len(data_loader)).item(), 10)}"
                )
                wandb.log(
                    {
                        f"Test loss {config.data_dirs_test[i][5:]}, {str(loss_module)}": total_convert_loss
                        / len(data_loader)
                    },
                    step=current_epoch,
                )

    return total_convert_loss.item() / len(data_loader)


def model_pipeline(
    hyperparameters, ndata_dict, loss_dict, optimizer_dict, mode_wandb, losses
):
    # tell wandb to get started
    with wandb.init(
        project="test", config=hyperparameters, mode=mode_wandb, tags=[str(device)]
    ):
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config
        wandb.run.name = f"{config.architecture}/{config.data_type}/{config.iter}"

        # make the model, data, and optimization problem
        model, train_loader, test_loader, criterion, optimizer = make(
            config, ndata_dict, loss_dict, optimizer_dict
        )
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
            losses,
        )

        # and test its final performance
        eval_model(model, test_loader, config, config.epochs, losses)

    return model


def make(config, ndata_dict, loss_dict, optimizer_dict):
    # Make the data
    data_set_train = MyDataset(
        sims=config.train_sims,
        n_frames=config.n_frames,
        n_data=ndata_dict[config.data_type],
        data_type=config.data_type,
        dir=config.data_dir_train,
    )
    # data_set_test = MyDataset(sims=config.test_sims, n_frames=config.n_frames, n_data=ndata_dict[config.data_type], data_type=config.data_type, dir=config.data_dir_train)

    train_data_loader = data.DataLoader(
        data_set_train, batch_size=config.batch_size, shuffle=True
    )

    print("-- Finished Train Dataloader --")
    # test_data_loader = data.DataLoader(data_set_test, batch_size=config.batch_size, shuffle=True, drop_last=False)

    test_data_loaders = []

    for test_data_dir in config.data_dirs_test:
        data_set_test = MyDataset(
            sims=config.test_sims,
            n_frames=config.n_frames,
            n_data=ndata_dict[config.data_type],
            data_type=config.data_type,
            dir="data/" + test_data_dir,
            # dir="data/"+test_data_dir #TODO Only for testing
        )
        test_data_loader = data.DataLoader(
            data_set_test, batch_size=config.batch_size, shuffle=True, drop_last=False
        )
        test_data_loaders += [test_data_loader]

    print("-- Finished Test Dataloader(s) --")

    # Make the model
    model = GRU(config, ndata_dict[config.data_type], num_outputs=3).to(device)

    # Make the loss and optimizer
    criterion = loss_dict[config.loss_type](reduction=config.loss_reduction_type)
    optimizer = optimizer_dict[config.optimizer](
        model.parameters(), lr=config.learning_rate
    )

    return model, train_data_loader, test_data_loaders, criterion, optimizer


if __name__ == "__main__":
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
        default="data_t(0, 0)_r(2, 5)_none_pNone_gNone",
    )
    parser.add_argument("-l", "--loss", type=str, help="Loss type", default="L2")
    parser.add_argument("--data_type", type=str, help="Type of data", default="pos")
    parser.add_argument(
        "-i", "--iterations", type=int, help="Number of iterations", default=1
    )
    parser.add_argument("--inertia_input", action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    data_dir_train = "data/" + " ".join(args.data_dir_train)
    # data_dirs_test = args.data_dir_test
    data_dirs_test = [os.listdir("data")[3]]  # TODO ONLY FOR TESTing
    if ".DS_Store" in data_dirs_test:
        data_dirs_test.remove(".DS_Store")
    data_dirs_test = [data_dir_train]
    data_dirs_test = [
        " ".join(args.data_dir_train),
        "data_t(0, 0)_tennisEffect",
    ]

    # if args.data_dir_test == "":
    #     data_dirs_test = [data_dir_train]
    # else:
    #     data_dirs_test = "data/" + args.data_dir_test

    losses = [nn.MSELoss]
    if not os.path.exists(data_dir_train):
        raise IndexError("No directory for the train data {args.data_dir_train}")

    for i in range(args.iterations):
        print(f"----- ITERATION {i+1}/{args.iterations} ------")
        # Divide the train en test dataset
        n_sims_train_total = len(os.listdir(data_dir_train))
        n_sims_train_total = 1000
        sims_train = {i for i in range(n_sims_train_total)}
        train_sims = set(random.sample(sims_train, int(0.8 * n_sims_train_total)))
        test_sims = sims_train - train_sims
        print(f"Number of train simulations: {len(train_sims)}")
        print(f"Number of test simulations: {len(test_sims)}")
        # if data_dir_train == data_dir_test:
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

        config = dict(
            learning_rate=0.005,
            epochs=30,
            batch_size=128,
            dropout=0.2,
            loss_type=args.loss,
            loss_reduction_type="mean",
            optimizer="Adam",
            data_type=args.data_type,
            architecture="gru",
            train_sims=list(train_sims),
            test_sims=list(test_sims),
            n_frames=30,
            n_sims=n_sims_train_total,
            n_layers=1,
            hidden_size=96,
            data_dir_train=data_dir_train,
            data_dirs_test=data_dirs_test,
            iter=i,
            inertia_input=args.inertia_input,
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
            "log_dualQ": 6,
        }

        start_time = time.time()
        model = model_pipeline(
            config, ndata_dict, loss_dict, optimizer_dict, args.mode_wandb, losses
        )
        print("It took ", time.time() - start_time, " seconds.")

        model_dict = {
            "config": config,
            "data_dict": ndata_dict,
            "model": model.state_dict(),
        }

        if not os.path.exists("models"):
            os.mkdir("models")

        torch.save(
            model_dict,
            f"models/lstm/{config['data_type']}_{config['architecture']}_'{args.data_dir_train}'.pickle",
        )
