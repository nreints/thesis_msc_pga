import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as data
from convert import *
import pickle
import random
import wandb

wandb.init(project="my-test-project")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class LSTM(nn.Module):
    def __init__(self, in_size, hidden_size, n_layers=2):
        super().__init__()
        # Initialize the modules we need to build the network
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.in_size = in_size
        self.lstm = nn.LSTM(in_size, hidden_size, batch_first=True)
        self.layers = nn.Sequential(
            nn.Linear(hidden_size, in_size)
        )

    def forward(self, x, hidden_state=None):
        # Perform the calculation of the model to determine the prediction
        # print(x.shape)

        batch_size, _, _ = x.shape
        if hidden_state == None:
            hidden_state = torch.zeros(self.n_layers, batch_size, self.hidden_size)
            cell_state = torch.zeros(self.n_layers, batch_size, self.hidden_size)
        else:
            hidden_state, cell_state = hidden_state
        # print(self.lstm(x, (hidden_state, cell_state))[0].shape)
        out, h = self.lstm(x)
        return self.layers(out), h



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

                    self.start_pos.append(data_all["pos"][0].reshape(-1, 24).squeeze())
                    train_end = frame + self.n_frames_perentry
                    self.data.append(data[frame:train_end].reshape(-1, self.n_datap_perframe))
                    self.target.append(data[frame+1:train_end+1].reshape(-1, self.n_datap_perframe))

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



def train_model(model, optimizer, data_loader, test_loader, loss_module, num_epochs=100, loss_type="L1"):
    # Set model to train mode
    model.train()

    # Training loop
    for epoch in range(num_epochs):
        loss_epoch = 0
        for data_inputs, data_labels, _ in data_loader:
            data_inputs = data_inputs.to(device)
            data_labels = data_labels.to(device)
            # print(data_inputs.shape)
            # print(data_labels.shape)
            # print(data_inputs[0].shape)
            # print(data_labels[0].reshape(-1, 20, 24))
            # print(epoch)
            # print(data_inputs.shape)
            # data_inputs = data_inputs.reshape(-1,20, 24) #????
            # data_labels = data_labels.reshape(-1, 20, 24)

            output, _ = model(data_inputs)

            # print(data_inputs[0])
            # print(data_labels.shape)
            # print(output[0, -1])
            # print(data_labels[0, -1])


            # print(output.squeeze().shape)
            # print(data_labels.shape)
            loss = loss_module(output.squeeze(), data_labels.float())
            # The gradients would not be overwritten, but actually added to the existing ones.
            wandb.log({"loss": loss})

            optimizer.zero_grad()
            # Perform backpropagation
            loss.backward()

            ## Step 5: Update the parameters
            optimizer.step()


        # print(epoch) /
            # exit()



            # ## Step 1: Move input data to device (only strictly necessary if we use GPU)
            # data_inputs = data_inputs.to(device)
            # data_labels = data_labels.to(device)

            # ## Step 2: Run the model on the input data
            # preds = model(data_inputs)
            # preds = preds.squeeze(dim=1) # Output is [Batch size, 1], but we want [Batch size]

            # ## Step 3: Calculate the loss
            # loss = loss_module(preds, data_labels)
            loss_epoch += loss

            # ## Step 4: Perform backpropagation
            # # Before calculating the gradients, we need to ensure that they are all zero.
            # # The gradients would not be overwritten, but actually added to the existing ones.
            # optimizer.zero_grad()
            # # Perform backpropagation
            # loss.backward()

            # ## Step 5: Update the parameters
            # optimizer.step()

        if epoch % 10 == 0:
            true_loss, convert_loss = eval_model(model, test_loader, loss_module)
            model.train()
            print(epoch, round(loss_epoch.item()/len(data_loader), 10), "\t", round(true_loss, 10), '\t', round(convert_loss, 10))

            # f = open(f"results/{data_type}/{num_epochs}_{lr}_{loss_type}.txt", "a")
            # f.write(f"{[epoch, round(loss_epoch.item()/len(data_loader), 10), round(true_loss, 10), round(convert_loss, 10)]} \n")
            # f.write("\n")
            # f.close()


def eval_model(model, data_loader, loss_module):
    # return 0, 0
    model.eval() # Set model to eval mode

    with torch.no_grad(): # Deactivate gradients for the following code
        total_loss = 0
        total_convert_loss = 0
        for data_inputs, data_labels, start_pos in data_loader:

            # Determine prediction of model on dev set
            data_inputs, data_labels = data_inputs.to(device), data_labels.to(device)

            preds, _ = model(data_inputs)
            # print(preds.shape)

            preds = preds.squeeze(dim=1)

            # print(preds.shape)

            alt_preds = convert(preds.detach().cpu(), start_pos, data_loader.dataset.data_type)
            alt_labels = convert(data_labels.detach().cpu(), start_pos, data_loader.dataset.data_type)

            total_loss += loss_module(preds, data_labels)
            total_convert_loss += loss_module(alt_preds, alt_labels)

    return total_loss.item()/len(data_loader), total_convert_loss.item()/len(data_loader)



n_frames = 20
n_sims = 750

data_type = "pos"
n_data = 24 # xyz * 8

# data_type = "eucl_motion"
# n_data = 12

# # data_type = "quat"
# # n_data = 7

# # data_type = "log_quat"
# # n_data = 7

# # data_type = "pos_diff"
# # n_data = 24

# data_type = "pos_diff_start"
# n_data = 24



sims = {i for i in range(n_sims)}
train_sims = set(random.sample(sims, int(0.8 * n_sims)))
test_sims = sims - train_sims
print("DATATYPE", data_type)

batch_size = 128

model = LSTM(n_data, 96)
model.to(device)

data_set_train = MyDataset(sims=train_sims, n_frames=n_frames, n_data=n_data, data_type=data_type)
data_set_test = MyDataset(sims=test_sims, n_frames=n_frames, n_data=n_data, data_type=data_type)

train_data_loader = data.DataLoader(data_set_train, batch_size=batch_size, shuffle=True)
test_data_loader = data.DataLoader(data_set_test, batch_size=batch_size, shuffle=True, drop_last=False)

# # exit()
# lrs = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
# for lr in lrs:
    # print("Testing lr ", lr, "Datatype ", data_type)
num_epochs = 400

loss = "L1"
wandb.config = {
    "loss_module": loss,
    # "learning_rate": lr,
    "epochs": num_epochs,
    "batch_size": batch_size
    }
loss_module = nn.L1Loss()

# f = open(f"results/{data_type}/{num_epochs}_{lr}_{loss}.txt", "w")
# f.write(f"Data type: {data_type}, num_epochs: {num_epochs}, \t lr: {lr} \n")

optimizer = torch.optim.Adam(model.parameters())

train_model(model, optimizer, train_data_loader, test_data_loader, loss_module, num_epochs=num_epochs, loss_type=loss)

test_data_loader = data.DataLoader(data_set_test, batch_size=batch_size, shuffle=False, drop_last=False)
eval_model(model, test_data_loader, loss_module)
print("-------------------------")