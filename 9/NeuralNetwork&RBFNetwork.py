import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster
import pandas as pd
import sklearn.datasets
import torch.nn as nn
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

# -----------------------------------------------------
# Part a
# -----------------------------------------------------


data = pd.read_csv("data.csv")

first_column = data.iloc[:, 0].to_numpy().reshape(-1,1) 
second_column = data.iloc[:, 1].to_numpy()  

def make_rbf(mu, sigma):
    def rbf(x):
        return np.exp(-np.linalg.norm(x - mu, axis=1)**2 / sigma**2)
    return rbf

cluster_centers = np.array([2,5,8])

phis = [make_rbf(mu, 1) for mu in cluster_centers]

features_new = np.column_stack([phi(first_column) for phi in phis])


xy_data = np.column_stack((first_column, second_column))

W,_,_,_ = np.linalg.lstsq(features_new,second_column, rcond=None)


def H(x):
    features = np.column_stack([phi(x) for phi in phis])
    return features @ W

x_plot = np.linspace(0, 10, 100).reshape(-1, 1)
y_plot = H(x_plot)

plt.scatter(first_column, second_column, label='Data')
plt.plot(x_plot, y_plot, color='red', label='RBF Network')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('RBF Network Fit to Data')
plt.show()


# ----------------------------------------------------
# Part b - bonus
# -----------------------------------------------------

# data = pd.read_csv("data.csv")

# first_column = data.iloc[:, 0].to_numpy().reshape(-1, 1)
# second_column = data.iloc[:, 1].to_numpy()

# X = torch.tensor(first_column, dtype=torch.float32)
# y = torch.tensor(second_column, dtype=torch.float32).unsqueeze(1)

# class NN_new(nn.Module):
#     def __init__(self, inp_nodes, hid_nodes, out_nodes):
#         super().__init__()
#         self.inp_dim = inp_nodes
#         self.hid_dim = hid_nodes
#         self.out_dim = out_nodes

#         self.inp_layer = nn.Linear(self.inp_dim, self.hid_dim)
#         self.hid_layer1 = nn.Linear(self.hid_dim, self.hid_dim)
#         self.hid_layer2 = nn.Linear(self.hid_dim, self.hid_dim)
#         self.out_layer = nn.Linear(self.hid_dim, self.out_dim)

#         self.sgm1 = nn.Sigmoid()
#         self.sgm2 = nn.Sigmoid()

#     def forward(self, x):
#         x = self.inp_layer(x)
#         x = self.sgm1(x)
#         x = self.hid_layer1(x)
#         x = self.sgm2(x)
#         x = self.hid_layer2(x)
#         x = self.out_layer(x)
#         return x

# def train_func(model, num_epochs, optimizer, loss_func, dataloader, logging=False):
#     model.train() 
#     for epoch in range(1, num_epochs + 1): 
#         epoch_loss = 0.0
#         for batch_X, batch_y in dataloader:
#             optimizer.zero_grad()
#             y_pred = model(batch_X)
#             loss_value = loss_func(y_pred, batch_y)
#             loss_value.backward()
#             optimizer.step()
#             epoch_loss += loss_value.item()
#         if logging:
#             print(f'Epoch {epoch}, Loss: {epoch_loss/len(dataloader)}')
#     return model

# dataset = TensorDataset(X, y)
# dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# model = NN_new(1, 5, 1)
# loss_func = nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# model = train_func(model, 10000, optimizer, loss_func, dataloader)

# model.eval()
# predictions = model(X).detach().numpy()

# plt.scatter(X.numpy(), y.numpy(), label='True Data')
# plt.scatter(X.numpy(), predictions, label='Predicted Data', c = predictions, cmap='RdYlGn')
# plt.xlabel('Feature')
# plt.ylabel('Target')
# plt.legend()
# plt.title('Model Predictions vs True Data')
# plt.show()