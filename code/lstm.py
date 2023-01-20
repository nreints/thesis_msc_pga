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

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class LSTM(nn.Module):
    def __init__(self, in_size, config):
        super().__init__()
        # Initialize the modules we need to build the network
        self.n_layers = config["n_layers"]
        self.hidden_size = config["hidden_size"]
        self.in_size = in_size
        self.lstm = nn.LSTM(in_size, self.hidden_size, batch_first=True, dropout=config["dropout"], num_layers=config["n_layers"])
        self.layers = nn.Sequential(
            nn.Linear(self.hidden_size, in_size)
        )

    def forward(self, x, hidden_cell=None):
        # Perform the calculation of the model to determine the prediction

        batch_size, _, _ = x.shape
        if hidden_cell == None:
            hidden_state = torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device)
            cell_state = torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device)
        else:
            hidden_state, cell_state = hidden_cell
        out, h = self.lstm(x, (hidden_state, cell_state))
        return self.layers(out), h


class MyDataset(data.Dataset):

    def __init__(self, sims, n_frames, n_data, data_type, dir):
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
        self.collect_data()

    def collect_data(self):
        self.data = []
        self.target = []
        self.target_pos = []
        self.start_pos = []

        for i in self.sims:
            with open(f'{self.dir}/sim_{i}.pickle', 'rb') as f:
                data_all = pickle.load(f)["data"]
                data = data_all[self.data_type]
                for frame in range(len(data) - (self.n_frames_perentry + 1)):
                    if self.data_type == "pos_diff_start":
                        self.start_pos.append(data_all["pos"][0].flatten())
                    else:
                        self.start_pos.append(data_all["start"].flatten())
                    train_end = frame + self.n_frames_perentry
                    self.data.append(data[frame:train_end].reshape(-1, self.n_datap_perframe))
                    self.target.append(data[frame+1:train_end+1].reshape(-1, self.n_datap_perframe))
                    self.target_pos.append(data_all["pos"][frame+1:train_end+1])


        self.data = torch.FloatTensor(np.asarray(self.data))
        self.target = torch.FloatTensor(np.asarray(self.target))
        self.target_pos = torch.FloatTensor(np.asarray(self.target_pos)).flatten(start_dim=2)
        self.start_pos = torch.FloatTensor(np.asarray(self.start_pos))

    def __len__(self):
        # Number of data point we have. Alternatively self.data.shape[0], or self.label.shape[0]
        return self.data.shape[0]

    def __getitem__(self, idx):
        # Return the idx-th data point of the dataset
        data_point = self.data[idx]
        data_target = self.target[idx]
        data_target_pos = self.target_pos[idx]
        data_start = self.start_pos[idx]
        return data_point, data_target, data_target_pos, data_start

def train_log(loss, epoch):
    wandb.log({"Epoch": epoch, "Train loss": loss}, step=epoch)
    # print(f"Loss after " + f" examples: {loss:.3f}")

def train_model(model, optimizer, data_loader, test_loader, loss_module, num_epochs, config):
    # Set model to train mode
    model.train()
    wandb.watch(model, loss_module, log="all", log_freq=10)

    # Training loop
    for epoch in range(num_epochs):
        loss_epoch = 0
        epoch_time = time.time()

        for data_inputs, data_labels, pos_target, start_pos in data_loader:
            # start = time.time()

            data_inputs = data_inputs.to(device)
            data_labels = data_labels.to(device)
            pos_target = pos_target.to(device)
            start_pos = start_pos.to(device)

            output, _ = model(data_inputs)
            # print(output.shape)
            # if config['data_type'] == 'pos':
            #     # print(output.shape, data_labels_pos.shape)
            #     output = output.reshape((output.shape[0], output.shape[1], 8, 3))


            # print("inputs", data_inputs.shape)
            # print("labels", data_labels.shape)
            # print("pos_target", pos_target.shape)

            alt_preds = convert(output, start_pos, data_loader.dataset.data_type)

            loss = loss_module(alt_preds, pos_target)

            optimizer.zero_grad()

            # Perform backpropagation
            loss.backward()

            ## Step 5: Update the parameters
            optimizer.step()

            loss_epoch += loss

            # print("total_time", time.time() - start)

        train_log(loss_epoch/len(data_loader), epoch)

        convert_loss = eval_model(model, test_loader, loss_module, config)
        model.train()
        print(epoch, round(loss_epoch.item()/len(data_loader), 10), '\t', round(convert_loss, 10))

        # f = open(f"results/{data_type}/{num_epochs}_{lr}_{loss_type}.txt", "a")
        # f.write(f"{[epoch, round(loss_epoch.item()/len(data_loader), 10), round(true_loss, 10), round(convert_loss, 10)]} \n")
        # f.write("\n")
        # f.close()
        print("epoch_time; ", time.time() - epoch_time)



def eval_model(model, data_loader, loss_module, config):
    model.eval() # Set model to eval mode

    with torch.no_grad(): # Deactivate gradients for the following code
        total_loss = 0
        total_convert_loss = 0
        for data_inputs, data_labels, data_labels_pos, start_pos in data_loader:

            # Determine prediction of model on dev set
            data_inputs, data_labels = data_inputs.to(device), data_labels.to(device)

            preds, _ = model(data_inputs)
            preds = preds.squeeze(dim=1)

            # if config['data_type'] == 'pos':
            #     preds = preds.reshape((preds.shape[0], preds.shape[1], 8, 3))

            alt_preds = convert(preds.detach().cpu(), start_pos, data_loader.dataset.data_type)

            total_loss += loss_module(preds, data_labels)
            total_convert_loss += loss_module(alt_preds, data_labels_pos)

        wandb.log({"Converted test loss": total_convert_loss/len(data_loader)})

    return total_convert_loss.item()/len(data_loader)



def model_pipeline(hyperparameters, ndata_dict, loss_dict, optimizer_dict, data_dir, mode_wandb):
    # tell wandb to get started
    with wandb.init(project="thesis", config=hyperparameters, mode=mode_wandb):
      # access all HPs through wandb.config, so logging matches execution!
      config = wandb.config
      wandb.run.name = f"{config.architecture}/{config.data_type}"

      # make the model, data, and optimization problem
      model, train_loader, test_loader, criterion, optimizer = make(config, ndata_dict, loss_dict, optimizer_dict, data_dir)
      print(model)

      # and use them to train the model
      train_model(model, optimizer, train_loader, test_loader, criterion, config.epochs, config)

      # and test its final performance
      eval_model(model, test_loader, criterion, config)

    return model

def make(config, ndata_dict, loss_dict, optimizer_dict, data_dir):
    # Make the data
    data_set_train = MyDataset(sims=config.train_sims, n_frames=config.n_frames, n_data=ndata_dict[config.data_type], data_type=config.data_type, dir=data_dir)
    data_set_test = MyDataset(sims=config.test_sims, n_frames=config.n_frames, n_data=ndata_dict[config.data_type], data_type=config.data_type, dir=data_dir)

    train_data_loader = data.DataLoader(data_set_train, batch_size=config.batch_size, shuffle=True)
    test_data_loader = data.DataLoader(data_set_test, batch_size=config.batch_size, shuffle=True, drop_last=False)

    # Make the model
    model = LSTM(ndata_dict[config.data_type], config).to(device)

    # Make the loss and optimizer
    criterion = loss_dict[config.loss_type](reduction=config.loss_reduction_type)
    optimizer = optimizer_dict[config.optimizer](
        model.parameters(), lr=config.learning_rate)

    return model, train_data_loader, test_data_loader, criterion, optimizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode_wandb", type=str, help="mode of wandb: online, offline, disabled", default="online")
    parser.add_argument("-data_dir_train", type=str, help="directory of the train data", default=f"data_t(0, 0)_r(0, 0)_none")
    parser.add_argument("-data_dir_test", type=str, help="directory of the test data", default="")
    args = parser.parse_args()
    data_dir_train = "data/" + args.data_dir_train
    data_dir_test = "data/" + args.data_dir_test

    if not os.path.exists(data_dir_train):
        raise("No directory for the train data {args.data_dir_train}")
    # TODO FIX DIFFERENT train test data
    # if not args.data_dir_test or not os.path.exists(args.data_dir_test):
    #     raise("No directory for the test data {args.data_dir_test}")
    print(data_dir_train)
    # for data_dir in args.data_dir_train:
    for data_thing in ["pos", "eucl_motion", "quat", "log_quat", "dual_quat", "pos_diff_start", "log_dualQ"]:
        n_sims = len(os.listdir(data_dir_train))
        sims = {i for i in range(n_sims)}
        train_sims = set(random.sample(sims, int(0.8 * n_sims)))
        test_sims = sims - train_sims

        config = dict(
            learning_rate = 0.005,
            epochs = 30,
            batch_size = 1024,
            dropout = 0,
            loss_type = "L1",
            loss_reduction_type = "mean",
            optimizer = "Adam",
            data_type = data_thing,
            architecture = "lstm",
            train_sims = list(train_sims),
            test_sims = list(test_sims),
            n_frames = 30,
            n_sims = n_sims,
            n_layers = 1,
            hidden_size = 96,
            data_dir=data_dir_train
            )

        loss_dict = {
                    'L1': nn.L1Loss,
                    'L2': nn.MSELoss
                    }

        optimizer_dict = {'Adam': torch.optim.Adam}

        ndata_dict = {
                        "pos": 24,
                        "eucl_motion": 12,
                        "quat": 7,
                        "log_quat": 7,
                        "dual_quat": 8,
                        "pos_diff": 24,
                        "pos_diff_start": 24,
                        "log_dualQ": 6
                    }
        start_time = time.time()
        print(config["data_type"])
        model = model_pipeline(config, ndata_dict, loss_dict, optimizer_dict, data_dir_train, args.mode_wandb)
        print("It took ", time.time() - start_time, " seconds.")

        model_dict = {'config': config,
                    'data_dict': ndata_dict,
                    'model': model.state_dict()}

        if not os.path.exists("models"):
            os.mkdir("models")


        torch.save(model_dict, f"models/lstm/{config['data_type']}_{config['architecture']}.pickle")