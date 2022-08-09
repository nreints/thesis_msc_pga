import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import torch.utils.data as data
import random
from convert import *
import wandb

# wandb.init(project="thesis_linearNN")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class Network(nn.Module):

    def __init__(self, n_data, config):
        super().__init__()
        # Initialize the modules we need to build the network
        self.layers = [nn.Linear(config.n_frames * n_data, config.hidden_sizes[0])]

        for i in range(len(config.hidden_sizes)):

            if config.batch_norm[i]:
                self.layers += [nn.BatchNorm1d(config.hidden_sizes[i])]

            if config.activation_func[i] == "Tanh":
                self.layers += [nn.Tanh()]
            elif config.activation_func[i] == "ReLU":
                self.layers += [nn.ReLU()]
            else:
                raise ValueError('Wrong activation func')

            self.layers += [nn.Dropout](p=config.dropout[i])

            if i < len(config.hidden_sizes) - 1:
                self.layers += [nn.Linear(config.hidden_sizes[i], config.hidden_sizes[i+1])]

        self.layers += [nn.Linear(config.hidden_sizes[-1], n_data)]

        self.linears = nn.ModuleList(self.layers)

        # self.layers = nn.Sequential(
        #     nn.Linear(n_steps * n_data, 256),
        #     nn.Tanh(),
        #     nn.Dropout(p=0.7),
        #     nn.Linear(256, n_out),
        #     # nn.Linear(n_out, 256),
        #     # nn.BatchNorm1d(256),
        #     # nn.Tanh(),
        #     # nn.Dropout(p=0.7),
        #     # nn.Linear(256, n_out),
        #     # nn.BatchNorm1d(n_hidden2),
        #     # nn.ReLU(),
        #     # nn.Dropout(p=0.3),
        #     # nn.Linear(n_hidden2, n_out)
        #     )

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
        # self.data = torch.empty((self.n_sims * , self.n_frames_perentry * self.n_datap_perframe))
        # self.target = torch.empty((1, self.n_datap_perframe))

        self.data = []
        self.target = []
        self.start_pos = []

        for i in self.sims:
            with open(f'data/sim_{i}.pickle', 'rb') as f:
                data_all = pickle.load(f)["data"]
                data = data_all[self.data_type]
                for frame in range(len(data) - (self.n_frames_perentry + 1)):
                    self.start_pos.append(data_all["pos"][0])
                    train_end = frame + self.n_frames_perentry
                    self.data.append(data[frame:train_end].flatten())
                    self.target.append(data[train_end+1].flatten())

        self.data = torch.FloatTensor(np.asarray(self.data))
        self.target = torch.FloatTensor(np.asarray(self.target))
        self.start_pos = torch.FloatTensor(np.asarray(self.start_pos))

    def __len__(self):
        # Number of data point we have. Alternatively self.data.shape[0], or self.label.shape[0]
        return self.data.shape[0]

    def __getitem__(self, idx):
        # Return the idx-th data point of the dataset
        # If we have multiple things to return (data point and label), we can return them as tuple
        data_point = self.data[idx]
        data_target = self.target[idx]
        data_start = self.start_pos[idx]
        return data_point, data_target, data_start


def train_log(loss, epoch):
    wandb.log({"epoch": epoch, "loss": loss}, step=epoch)
    print(f"Loss after " + f" examples: {loss:.3f}")

def train_model(model, optimizer, data_loader, test_loader, loss_module, num_epochs=100, loss_type="L1"):
    # Set model to train mode
    model.train()
    wandb.watch(model, loss_module, log="all", log_freq=10)

    # Training loop
    for epoch in range(num_epochs):
        loss_epoch = 0
        for data_inputs, data_labels, start_pos in data_loader:

            ## Step 1: Move input data to device (only strictly necessary if we use GPU)
            data_inputs = data_inputs.to(device)
            data_labels = data_labels.to(device)

            ## Step 2: Run the model on the input data
            preds = model(data_inputs)
            preds = preds.squeeze(dim=1) # Output is [Batch size, 1], but we want [Batch size]

            # print(data_inputs[0].reshape(-1, 24))

            # print(data_labels[0])
            # print(preds[0])

            ## Step 3: Calculate the loss

            # alt_preds = convert(preds, start_pos, data_loader.dataset.data_type)
            # alt_labels = convert(data_labels, start_pos, data_loader.dataset.data_type)
            # loss = loss_module(alt_preds, alt_labels)

            loss = loss_module(preds, data_labels)
            loss_epoch += loss


            # Optional
            wandb.watch(model)

            ## Step 4: Perform backpropagation
            # Before calculating the gradients, we need to ensure that they are all zero.
            # The gradients would not be overwritten, but actually added to the existing ones.
            optimizer.zero_grad()
            # Perform backpropagation
            loss.backward()

            ## Step 5: Update the parameters
            optimizer.step()

        train_log(loss_epoch/len(data_loader), epoch)
        if epoch % 10 == 0:
            true_loss, convert_loss = eval_model(model, test_loader, loss_module)
            model.train()
            print(epoch, round(loss_epoch.item()/len(data_loader), 10), "\t", round(true_loss, 10), '\t', round(convert_loss, 10))

            f = open(f"results/{data_type}/{num_epochs}_{lr}_{loss_type}.txt", "a")
            f.write(f"{[epoch, round(loss_epoch.item()/len(data_loader), 10), round(true_loss, 10), round(convert_loss, 10)]} \n")
            f.write("\n")
            f.close()



def eval_model(model, data_loader, loss_module):
    model.eval() # Set model to eval mode

    with torch.no_grad(): # Deactivate gradients for the following code
        total_loss = 0
        total_convert_loss = 0
        for data_inputs, data_labels, start_pos in data_loader:

            # Determine prediction of model on dev set
            data_inputs, data_labels = data_inputs.to(device), data_labels.to(device)
            preds = model(data_inputs)
            preds = preds.squeeze(dim=1)

            alt_preds = convert(preds.detach().cpu(), start_pos, data_loader.dataset.data_type)
            alt_labels = convert(data_labels.detach().cpu(), start_pos, data_loader.dataset.data_type)

            # print(alt_preds.shape, alt_labels.shape)
            total_loss += loss_module(preds, data_labels)
            total_convert_loss += loss_module(alt_preds, alt_labels)

            wandb.log({"total converted loss": total_convert_loss/len(data_loader)})

    return total_loss.item()/len(data_loader), total_convert_loss.item()/len(data_loader)


n_frames = 20
n_sims = 750

data_type = "pos"
n_data = 24 # xyz * 8

# data_type = "eucl_motion"
# n_data = 12

# data_type = "quat"
# n_data = 7

# data_type = "log_quat"
# n_data = 7

# data_type = "pos_diff"
# n_data = 24

# data_type = "pos_diff_start"
# n_data = 24

# sims = {i for i in range(n_sims)}
# train_sims = set(random.sample(sims, int(0.8 * n_sims)))
# test_sims = sims - train_sims

# batch_size = 128

# model = Network(n_frames, n_data, n_hidden1=96, n_hidden2=48, n_out=n_data)
# model.to(device)

# data_set_train = MyDataset(sims=train_sims, n_frames=n_frames, n_data=n_data, data_type=data_type)
# data_set_test = MyDataset(sims=test_sims, n_frames=n_frames, n_data=n_data, data_type=data_type)

# train_data_loader = data.DataLoader(data_set_train, batch_size=batch_size, shuffle=True)
# test_data_loader = data.DataLoader(data_set_test, batch_size=batch_size, shuffle=True, drop_last=False)


# epochs = 400
# reduction_type = "mean"
# optimizer_type = "Adam"
# # exit()
# lrs = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
# for lr in lrs:
#     loss = "L1"
#     wandb.config = {
#         "learning_rate": lr,
#         "epochs": epochs,
#         "batch_size": batch_size,
#         "loss_type": loss,
#         "loss_reduction_type": reduction_type,
#         "optimizer": optimizer_type,
#         "data_type": data_type,
#         "architecture": "fcnn",
#         }
#     print("Testing lr ", lr, "Datatype ", data_type)
#     num_epochs = epochs

#     loss_dict = {'L1': nn.L1Loss, 
#                 'L2': nn.MSELoss}

#     loss_module = loss_dict[loss](reduction=reduction_type)

#     # if loss=='L1':
#     #     loss_module = nn.L1Loss(reduction=reduction_type)

#     f = open(f"results/{data_type}/{num_epochs}_{lr}_{loss}.txt", "w")
#     f.write(f"Data type: {data_type}, num_epochs: {num_epochs}, \t lr: {lr} \n")

#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#     train_model(model, optimizer, train_data_loader, test_data_loader, loss_module, num_epochs=num_epochs, loss_type=loss)

#     test_data_loader = data.DataLoader(data_set_test, batch_size=batch_size, shuffle=False, drop_last=False)
#     eval_model(model, test_data_loader, loss_module)
#     print("-------------------------")


# ----------------------------
# ----------------------------

# n_frames = 20
n_sims = 750
sims = {i for i in range(n_sims)}
train_sims = set(random.sample(sims, int(0.8 * n_sims)))
test_sims = sims - train_sims



config = dict(
    learning_rate = 0.01,
    epochs = 400,
    batch_size = 128,
    loss_type = "L1",
    loss_reduction_type = "mean",
    optimizer = "Adam",
    data_type = "pos",
    architecture = "fcnn",
    train_sims = list(train_sims),
    test_sims = list(test_sims),
    n_frames = 20,
    n_sims = n_sims,
    hidden_sizes = [128, 256, 128],
    activation_func = ["Tanh", "Tanh", "ReLU"],
    dropout = [0.7, 0.8, 0.2],
    batch_norm = [True, True, True]
    )

loss_dict = {'L1': nn.L1Loss,
                'L2': nn.MSELoss}

optimizer_dict = {'Adam': torch.optim.Adam}

ndata_dict = {"pos": 24,
                "eucl_motion": 12,
                "quat": 7,
                "log_quat": 7,
                "pos_diff": 24,
                "pos_diff_start": 24,
            }

def model_pipeline(hyperparameters, ndata_dict, loss_dict, optimizer_dict):
    # print("hyperparams", hyperparameters)
    # tell wandb to get started
    with wandb.init(project="thesis", config=hyperparameters):
      # access all HPs through wandb.config, so logging matches execution!
      config = wandb.config
    #   print("model_pipeline", config)

      # make the model, data, and optimization problem
      model, train_loader, test_loader, criterion, optimizer = make(config, ndata_dict, loss_dict, optimizer_dict)
      print(model)

      # and use them to train the model
      train_model(model, train_loader, criterion, optimizer, config)

      # and test its final performance
      eval_model(model, test_loader)

    return model

def make(config, ndata_dict, loss_dict, optimizer_dict):
    # print("make",config)
    # Make the data
    data_set_train = MyDataset(sims=config.train_sims, n_frames=config.n_frames, n_data=ndata_dict[config.data_type], data_type=config.data_type)
    data_set_test = MyDataset(sims=config.test_sims, n_frames=config.n_frames, n_data=ndata_dict[config.data_type], data_type=config.data_type)

    train_data_loader = data.DataLoader(data_set_train, batch_size=config.batch_size, shuffle=True)
    test_data_loader = data.DataLoader(data_set_test, batch_size=config.batch_size, shuffle=True, drop_last=False)


    # Make the model
    model = Network(ndata_dict[config.data_type], config).to(device)

    # Make the loss and optimizer
    criterion = loss_dict[config.loss](reduction=config.reduction_type)
    optimizer = optimizer_dict[config.optimizer](
        model.parameters(), lr=config.learning_rate)

    return model, train_data_loader, test_data_loader, criterion, optimizer

model = model_pipeline(config, ndata_dict, loss_dict, optimizer_dict)