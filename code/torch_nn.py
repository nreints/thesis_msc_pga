import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import torch.utils.data as data
import random
from new_mujoco import rotVecQuat


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class Network(nn.Module):

    def __init__(self, n_steps, n_data, n_hidden1, n_hidden2, n_out):
        super().__init__()
        # Initialize the modules we need to build the network
        self.layers = nn.Sequential(
            nn.Linear(n_steps * n_data, 256),
            nn.BatchNorm1d(256),
            nn.Tanh(),
            nn.Dropout(p=0.3),
            nn.Linear(256, n_out),
            # nn.BatchNorm1d(n_hidden2),
            # nn.Tanh(),
            # nn.Dropout(p=0.3),
            # nn.Linear(n_hidden2, n_out)
            )

    def forward(self, x):
        # Perform the calculation of the model to determine the prediction
        return self.layers(x)



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
    #     self.collect_begin_pos_data()

    # def collect_begin_pos_data(self):
    #     self.begin_pos_data = []
    #     for i in range(self.sims):
    #         with open(f'data/pos/sim_{i}.pickle', 'rb') as f:
    #             data = pickle.load(f)["data"][0]
    #             self.begin_pos_data.append(data)

    #     self.begin_pos_data = torch.FloatTensor(np.asarray(self.begin_pos_data))

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


def train_model(model, optimizer, data_loader, test_loader, loss_module, num_epochs=100, loss_type="L1"):
    # Set model to train mode
    model.train()
    # print(data_loader.dataset.data_type)

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
            

            print(data_inputs[0].reshape(-1, 7))

            # for i in range(data_inputs.shape[0]):
            #     tmp = data_inputs[i].reshape(-1, 24)
            #     for j in range(24):
            #         if (j+1)%3!=0:
            #             check = torch.all(torch.eq(tmp[:, 3], tmp[0,3]))
            #             if not check:
            #                 print('BAD')
            #                 print(check)
            #                 exit()

            # print(data_labels[0])
            print(preds[0])

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

        if epoch % 10 == 0:
            true_loss, convert_loss = eval_model(model, test_loader, loss_module)
            model.train()
            print(epoch, round(loss_epoch.item()/len(data_loader), 10), "\t", round(true_loss, 10), '\t', round(convert_loss, 10))

            f = open(f"results/{data_type}/{num_epochs}_{lr}_{loss_type}.txt", "a")
            f.write(f"{[epoch, round(loss_epoch.item()/len(data_loader), 10), round(true_loss, 10), round(convert_loss, 10)]} \n")
            f.write("\n")
            f.close()

def eucl2pos(eucl_motion, start_pos):
    out = np.empty_like(start_pos)

    eucl_motion = eucl_motion.numpy().astype('float64')
    start_pos = start_pos.numpy().astype('float64')
    for batch in range(out.shape[0]):
        out[batch] =  (eucl_motion[batch,:9].reshape(3,3) @ start_pos[batch].T + np.vstack([eucl_motion[batch, 9:]]*8).T).T

    return torch.from_numpy(out.reshape((out.shape[0], -1)))


def quat2pos(quat, start_pos):
    out = np.empty_like(start_pos)

    if not isinstance(quat, np.ndarray):
        quat = quat.numpy().astype('float64')
    if not isinstance(start_pos, np.ndarray):
        start_pos = start_pos.numpy().astype('float64')
    for batch in range(out.shape[0]):
        for vert in range(out.shape[1]):
            out[batch, vert] = rotVecQuat(start_pos[batch,vert,:], quat[batch, :4]) + quat[batch, 4:]

    return torch.from_numpy(out.reshape((out.shape[0], -1)))

def log_quat2pos(log_quat, start_pos):

    log_quat = log_quat.numpy().astype('float64')
    start_pos = start_pos.numpy().astype('float64')
    rot_vec = log_quat[:, :3]
    angle = log_quat[:,3]
    trans = log_quat[:,4:]
    cos = np.cos(angle/2).reshape(-1, 1)
    sin = np.sin(angle/2)
    part1_1 = rot_vec * np.vstack([sin]*3).T
    part1 = np.append(cos, part1_1, axis=1)
    quat = np.append(part1, trans, axis=1)

    return quat2pos(quat, start_pos)

def convert(true_preds, start_pos, data_type):
    if data_type == "pos":
        return true_preds
    elif data_type == "eucl_motion":
        eucl2pos(true_preds, start_pos)
        return eucl2pos(true_preds, start_pos)
    elif data_type == "quat":
        return quat2pos(true_preds, start_pos)
    elif data_type == "log_quat":
        return log_quat2pos(true_preds, start_pos)
    return True


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
            # print(data_inputs[0].reshape(-1, 24))
            # print(data_labels[0])
            # print(preds[0])
            alt_labels = convert(data_labels.detach().cpu(), start_pos, data_loader.dataset.data_type)

            total_loss += loss_module(preds, data_labels)
            total_convert_loss += loss_module(alt_preds, alt_labels)

    return total_loss.item()/len(data_loader), total_convert_loss.item()/len(data_loader)



n_frames = 20
n_sims = 500

data_type = "pos"
n_data = 24 # xyz * 8

# data_type = "eucl_motion"
# n_data = 12

data_type = "quat"
n_data = 7

# data_type = "log_quat"
# n_data = 7

sims = {i for i in range(n_sims)}
train_sims = set(random.sample(sims, int(0.8 * n_sims)))
test_sims = sims - train_sims


model = Network(n_frames, n_data, n_hidden1=96, n_hidden2=48, n_out=n_data)
model.to(device)

data_set_train = MyDataset(sims=train_sims, n_frames=n_frames, n_data=n_data, data_type=data_type)
data_set_test = MyDataset(sims=test_sims, n_frames=n_frames, n_data=n_data, data_type=data_type)

train_data_loader = data.DataLoader(data_set_train, batch_size=128, shuffle=True)
test_data_loader = data.DataLoader(data_set_test, batch_size=128, shuffle=True, drop_last=False)

# exit()
lrs = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
for lr in lrs[:2]:
    print("Testing lr ", lr, "Datatype ", data_type)
    num_epochs = 1000

    loss = "L1"
    loss_module = nn.L1Loss(reduction='sum')

    f = open(f"results/{data_type}/{num_epochs}_{lr}_{loss}.txt", "w")
    f.write(f"Data type: {data_type}, num_epochs: {num_epochs}, \t lr: {lr} \n")

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    train_model(model, optimizer, train_data_loader, test_data_loader, loss_module, num_epochs=num_epochs, loss_type=loss)

    test_data_loader = data.DataLoader(data_set_test, batch_size=128, shuffle=False, drop_last=False)
    eval_model(model, test_data_loader, loss_module)
    print("-------------------------")