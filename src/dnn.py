import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from acopf import get_input_variables, get_output_variables


class MLP(nn.Module):
    """
    Standard MLP with variable hidden layers.
    """
    def __init__(self, input_dim, hidden_dim, num_hidden_layers, output_dim):
        super(MLP, self).__init__()
        self.activation = nn.ReLU()
        self.nhidden = num_hidden_layers
        self.fclayers = nn.ModuleList([nn.Linear(input_dim, hidden_dim)])
        for _ in range(num_hidden_layers-1):
            self.fclayers.append(nn.Linear(hidden_dim, hidden_dim))
        self.fclayers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        for d in range(self.nhidden):
            x = self.fclayers[d](x)
            x = self.activation(x)
        x = self.fclayers[self.nhidden](x)
        return x


class AdvancedMLP(nn.Module):
    """
    Advanced MLP suitable for deeper architectures.
    Includes Batch normalization and dropout to improve
    stability and generalization.
    """
    def __init__(self, input_dim, hidden_dim,
                 num_hidden_layers, output_dim, drop_rate=0.5):
        super(AdvancedMLP, self).__init__()
        self.activation = nn.ReLU()
        self.nhidden = num_hidden_layers
        self.fclayers = nn.ModuleList([nn.Linear(input_dim, hidden_dim)])
        self.dropp = drop_rate
        self.bnlayers = nn.ModuleList([nn.BatchNorm1d(hidden_dim)])
        self.dropout = nn.Dropout(self.dropp)

        for _ in range(num_hidden_layers-1):
            self.fclayers.append(nn.Linear(hidden_dim, hidden_dim))
            self.bnlayers.append(nn.BatchNorm1d(hidden_dim))
        self.fclayers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):

        for i in range(self.nhidden):
            x = self.fclayers[i](x)
            x = self.bnlayers[i](x)
            x = self.activation(x)
        x = self.dropout(x)
        x = self.fclayers[self.nhidden](x)
        return x



class TfData(Dataset):
    """
    For dataloading in torch.
    """
    def __init__(self,X_train, Y_train):
        self.num_samples = X_train.shape[0]
        self.X = torch.tensor(X_train, dtype=torch.float32)
        self.Y = torch.tensor(Y_train, dtype=torch.float32)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.X[idx, :], self.Y[idx, :]

def getcosts(opf_data):
    qc = torch.tensor(np.asarray(opf_data.gen_cost.q))
    lc = torch.tensor(np.asarray(opf_data.gen_cost.l))
    cc = torch.tensor(np.asarray(opf_data.gen_cost.c))
    return qc, lc, cc

def opf_cost(Y,opf_data, qc, lc, cc):
    pg, _, _, _ = get_output_variables(Y, opf_data)
    cost =   (qc* pg**2).sum(axis=1) + (lc* pg).sum(axis=1) + (torch.ones(pg.shape[0]))*sum(cc) 
    return cost

def equality_violations(X,Y, opf_data):
    pg, qg, vm, va = get_output_variables(Y, opf_data)
    pd, qd = get_input_variables(X, opf_data)
    # voltage shape: (num_samples * num_buses)
    voltage = vm * (torch.cos(va) + 1j * torch.sin(va))
    v_t = voltage.transpose(0, 1)
    y_bus = opf_data.y_bus.astype(np.complex64)
    y_bus = torch.sparse_coo_tensor(y_bus.nonzero(), y_bus.data, y_bus.shape)

    bus_injection = torch.multiply(voltage, torch.conj(
                   torch.mm(voltage, y_bus.transpose(0, 1))))
    generation = torch.zeros(
        (pg.shape[0], opf_data.get_num_buses()), dtype=torch.complex64)
    generation[:, opf_data.gen_bus_idx] = (pg + 1j*qg)[:, :]

    load = torch.zeros(
        (pg.shape[0], opf_data.get_num_buses()), dtype=torch.complex64)
    load[:, opf_data.load_bus_idx] = (pd + 1j*qd)[:, :]

    residual = generation - load - bus_injection
    temp = torch.concatenate([torch.real(residual), torch.imag(residual)], axis = 1)
    return temp


    
