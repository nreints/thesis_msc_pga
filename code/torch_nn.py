import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import torch.utils.data as data

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



WHY = 20

class SimpleClassifier(nn.Module):

    def __init__(self, n_steps, n_data, n_hidden1, n_hidden2, n_out):
        super().__init__()
        # Initialize the modules we need to build the network
        self.layers = nn.Sequential(
            nn.Linear(n_steps * n_data, n_hidden1),
            nn.ReLU(),
            nn.Linear(n_hidden1, n_hidden2),
            nn.ReLU(),
            nn.Linear(n_hidden2, n_out),
            )

    def forward(self, x):
        # Perform the calculation of the model to determine the prediction
        return self.layers(x)



class MyDataset(data.Dataset):

    def __init__(self, n_sims, n_frames, n_data):
        """
        Inputs:
            size - Number of data points we want to generate
            std - Standard deviation of the noise (see generate_continuous_xor function)
        """
        super().__init__()
        self.n_frames = n_frames
        self.n_data = n_data
        self.n_sims = n_sims
        self.collect_data()
        print(self.n_sims)


    def collect_data(self):
        self.data = torch.empty((self.n_sims, self.n_frames*self.n_data))
        self.target = torch.empty((1, self.n_data))

        for i in range(self.n_sims):
            print("i",i)
            with open(f'data/sim_{i}.pickle', 'rb') as f:
                data = pickle.load(f)["data"]
                print(len(data))
                for j in range(0, len(data)-8, self.n_frames*8):
                    print(j, j+self.n_frames*8)
                    print(data[j:j+self.n_frames*8].shape)
                    self.data[i] = torch.from_numpy(data[j:j+self.n_frames*8].flatten())
                    # self.target[i] = torch.from_numpy(data[(self.n_frames+1)*8:self.n_frames+9].flatten())

    def __len__(self):
        # Number of data point we have. Alternatively self.data.shape[0], or self.label.shape[0]
        return self.n_sims

    def __getitem__(self, idx):
        # Return the idx-th data point of the dataset
        # If we have multiple things to return (data point and label), we can return them as tuple
        data_point = self.data[idx]
        data_target = self.target[idx]
        return data_point, data_target



def train_model(model, optimizer, data_loader, loss_module, num_epochs=100):
    # Set model to train mode
    model.train()

    # Training loop
    for epoch in tqdm(range(num_epochs)):
        for data_inputs, data_labels in data_loader:

            ## Step 1: Move input data to device (only strictly necessary if we use GPU)
            data_inputs = data_inputs.to(device)
            data_labels = data_labels.to(device)

            ## Step 2: Run the model on the input data
            preds = model(data_inputs)
            preds = preds.squeeze(dim=1) # Output is [Batch size, 1], but we want [Batch size]

            ## Step 3: Calculate the loss
            loss = loss_module(preds, data_labels.float())

            ## Step 4: Perform backpropagation
            # Before calculating the gradients, we need to ensure that they are all zero. 
            # The gradients would not be overwritten, but actually added to the existing ones.
            optimizer.zero_grad()
            # Perform backpropagation
            loss.backward()

            ## Step 5: Update the parameters
            optimizer.step()

def eval_model(model, data_loader):
    model.eval() # Set model to eval mode
    true_preds, num_preds = 0., 0.

    with torch.no_grad(): # Deactivate gradients for the following code
        for data_inputs, data_labels in data_loader:

            # Determine prediction of model on dev set
            data_inputs, data_labels = data_inputs.to(device), data_labels.to(device)
            preds = model(data_inputs)
            preds = preds.squeeze(dim=1)
            preds = torch.sigmoid(preds) # Sigmoid to map predictions between 0 and 1
            pred_labels = (preds >= 0.5).long() # Binarize predictions to 0 and 1

            # Keep records of predictions for the accuracy metric (true_preds=TP+TN, num_preds=TP+TN+FP+FN)
            true_preds += (pred_labels == data_labels).sum()
            num_preds += data_labels.shape[0]

    acc = true_preds / num_preds
    print(f"Accuracy of the model: {100.0*acc:4.2f}%")

n_data = 24 # xyz * 8
n_framess = 20

model = SimpleClassifier(n_framess, n_data, n_hidden1=100, n_hidden2=60, n_out=24)


train_dataset = MyDataset(n_sims=10, n_frames=n_framess, n_data=n_data)
train_data_loader = data.DataLoader(train_dataset, batch_size=128, shuffle=True)

# model.to(device)

# loss_module = nn.BCEWithLogitsLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
# train_model(model, optimizer, train_data_loader, loss_module)


# test_dataset = MyDataset(size=500)
# test_data_loader = data.DataLoader(test_dataset, batch_size=128, shuffle=False, drop_last=False) 
# eval_model(model, test_data_loader)