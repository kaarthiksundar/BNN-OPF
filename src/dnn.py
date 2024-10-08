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

def convert_bounds_to_torch(opf_data):
    # Convert pg_bounds
    pg_lower = torch.tensor(np.asarray(opf_data.pg_bounds.lower), dtype=torch.float32)
    pg_upper = torch.tensor(np.asarray(opf_data.pg_bounds.upper), dtype=torch.float32)

    # Convert qg_bounds
    qg_lower = torch.tensor(np.asarray(opf_data.qg_bounds.lower), dtype=torch.float32)
    qg_upper = torch.tensor(np.asarray(opf_data.qg_bounds.upper), dtype=torch.float32)

    # Convert vm_bounds
    vm_lower = torch.tensor(np.asarray(opf_data.vm_bounds.lower), dtype=torch.float32)
    vm_upper = torch.tensor(np.asarray(opf_data.vm_bounds.upper), dtype=torch.float32)

    # Return all bounds as a dictionary or individual tensors based on need
    return {
        'pg_bounds': {'lower': pg_lower, 'upper': pg_upper},
        'qg_bounds': {'lower': qg_lower, 'upper': qg_upper},
        'vm_bounds': {'lower': vm_lower, 'upper': vm_upper}
    }

def pg_bound_violations_torch(pg, bounds):
    pg_lower = torch.maximum(bounds['pg_bounds']['lower'] - pg, torch.tensor(0.0))
    pg_upper = torch.maximum(pg - bounds['pg_bounds']['upper'], torch.tensor(0.0))
    return pg_lower, pg_upper

def qg_bound_violations_torch(qg, bounds):
    qg_lower = torch.maximum(bounds['qg_bounds']['lower'] - qg, torch.tensor(0.0))
    qg_upper = torch.maximum(qg - bounds['qg_bounds']['upper'], torch.tensor(0.0))
    return qg_lower, qg_upper

def vm_bound_violations_torch(vm, bounds):
    vm_lower = torch.maximum(bounds['vm_bounds']['lower'] - vm, torch.tensor(0.0))
    vm_upper = torch.maximum(vm - bounds['vm_bounds']['upper'], torch.tensor(0.0))
    return vm_lower, vm_upper

def inequality_constraint_violations_torch(Y, opf_data, bounds, line_limits=False):
    pg, qg, vm, va = get_output_variables(Y, opf_data)
    pg_lower, pg_upper = pg_bound_violations_torch(pg, bounds)
    qg_lower, qg_upper = qg_bound_violations_torch(qg, bounds)
    vm_lower, vm_upper = vm_bound_violations_torch(vm, bounds)
    if not line_limits:
        residual = torch.cat([
            pg_lower, pg_upper,
            qg_lower, qg_upper,
            vm_lower, vm_upper
        ], dim=1)
        return residual




