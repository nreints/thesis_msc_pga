import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import torch.utils.data as data
import random


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



WHY = 20

class Network(nn.Module):

    def __init__(self, n_steps, n_data, n_hidden1, n_hidden2, n_out):
        super().__init__()
        # Initialize the modules we need to build the network
        self.layers = nn.Sequential(
            nn.Linear(n_steps * n_data, n_hidden1),
            nn.Tanh(),
            nn.Linear(n_hidden1, n_hidden2),
            nn.Tanh(),
            nn.Linear(n_hidden2, n_out),
            )

    def forward(self, x):
        # Perform the calculation of the model to determine the prediction
        return self.layers(x)



class MyDataset(data.Dataset):

    def __init__(self, sims, n_frames, n_data):
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
        self.collect_data()


    def collect_data(self):
        # self.data = torch.empty((self.n_sims * , self.n_frames_perentry * self.n_datap_perframe))
        # self.target = torch.empty((1, self.n_datap_perframe))

        self.data = []
        self.target = []

        for i in self.sims:
            with open(f'data/sim_{i}.pickle', 'rb') as f:
                data = pickle.load(f)["data"]

                for frame in range(len(data) - (self.n_frames_perentry + 1)):
                    train_end = frame + self.n_frames_perentry
                    self.data.append(data[frame:train_end].flatten())
                    self.target.append(data[train_end+1].flatten())

        self.data = torch.FloatTensor(np.asarray(self.data))
        self.target = torch.FloatTensor(np.asarray(self.target))
        # print((self.data).shape, self.target.shape)

    def __len__(self):
        # Number of data point we have. Alternatively self.data.shape[0], or self.label.shape[0]
        return self.data.shape[0]

    def __getitem__(self, idx):
        # Return the idx-th data point of the dataset
        # If we have multiple things to return (data point and label), we can return them as tuple
        data_point = self.data[idx]
        data_target = self.target[idx]
        return data_point, data_target



def train_model(model, optimizer, data_loader, test_loader, loss_module, num_epochs=100):
    # Set model to train mode
    model.train()

    # Training loop
    for epoch in range(num_epochs):
        loss_epoch = 0
        for data_inputs, data_labels in data_loader:

            ## Step 1: Move input data to device (only strictly necessary if we use GPU)
            data_inputs = data_inputs.to(device)
            data_labels = data_labels.to(device)

            ## Step 2: Run the model on the input data
            preds = model(data_inputs)
            preds = preds.squeeze(dim=1) # Output is [Batch size, 1], but we want [Batch size]


            ## Step 3: Calculate the loss
            loss = loss_module(preds, data_labels)
            loss_epoch += loss

            ## Step 4: Perform backpropagation
            # Before calculating the gradients, we need to ensure that they are all zero.
            # The gradients would not be overwritten, but actually added to the existing ones.
            optimizer.zero_grad()
            # Perform backpropagation
            loss.backward()

            ## Step 5: Update the parameters
            optimizer.step()
        print(epoch, loss_epoch.item()/len(data_loader), "\t", eval_model(model, test_loader, loss_module))


def eval_model(model, data_loader, loss_module):
    model.eval() # Set model to eval mode

    with torch.no_grad(): # Deactivate gradients for the following code
        total_loss = 0
        for data_inputs, data_labels in data_loader:

            # Determine prediction of model on dev set
            data_inputs, data_labels = data_inputs.to(device), data_labels.to(device)
            preds = model(data_inputs)
            preds = preds.squeeze(dim=1)
            total_loss += loss_module(preds, data_labels)

    return total_loss.item()/len(data_loader)


n_data = 24 # xyz * 8
n_frames = 2
n_sims = 100

sims = {i for i in range(n_sims)}
train_sims = set(random.sample(sims, int(0.8 * n_sims)))
test_sims = sims - train_sims

model = Network(n_frames, n_data, n_hidden1=100, n_hidden2=60, n_out=24)


data_set_train = MyDataset(sims=train_sims, n_frames=n_frames, n_data=n_data)

data_set_test = MyDataset(sims=test_sims, n_frames=n_frames, n_data=n_data)

# print(data_set.data.shape)
# train_data_set = data_set[]
train_data_loader = data.DataLoader(data_set_train, batch_size=128, shuffle=True)

test_data_loader = data.DataLoader(data_set_test, batch_size=128, shuffle=False, drop_last=False)


model.to(device)

loss_module = nn.L1Loss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
train_model(model, optimizer, train_data_loader, test_data_loader, loss_module, num_epochs=500)

test_data_loader = data.DataLoader(data_set_test, batch_size=128, shuffle=False, drop_last=False) 
eval_model(model, test_data_loader, loss_module)