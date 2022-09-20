import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch_nn import Network

model = torch.load(f"models/pos_fcnn.pickle")
# print(model)

model_dict = torch.load(f"models/pos_fcnn.pickle")
config = model_dict['config']
ndata_dict = model_dict['data_dict']
model = Network(ndata_dict[config['data_type']], config)


# model.load_state_dict(torch.load(PATH))
