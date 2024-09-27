import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


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
