from abc import ABC
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam


class ExperimentBinaryModule(nn.Module):
    def __init__(self, input_dim, hidden_neurons_num=256):
        super(ExperimentBinaryModule, self).__init__()
        self.linear_1 = nn.Linear(input_dim, hidden_neurons_num)
        self.linear_2 = nn.Linear(hidden_neurons_num, 1)

    def forward(self, x):
        temp = F.relu(self.linear_1(x))
        return self.linear_2(temp)


class Module(ABC):
    def __init__(self, hyperparameters):
        self.model = None
        self.hyperparameters = hyperparameters

    def fit(self, X, y):
        pass

    def get_parameters(self):
        pass

    def set_parameters(self, parameters):
        pass

    def predict(self, X):
        pass

    def predict_proba(self, X, is_tensor=False, is_eval=False):
        pass


class MLPModule(Module):
    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)
        self.model = ((ExperimentBinaryModule(self.hyperparameters['input_dim'],
                                              self.hyperparameters['hidden_neurons_num']))
                      .to(self.hyperparameters['device']))
        self.optimizer = Adam(self.model.parameters(), lr=self.hyperparameters['lr'])
        self.criterion = nn.BCEWithLogitsLoss(reduction="none").to(self.hyperparameters['device'])

    def fit(self, X, y, weights=None):
        if weights is None:
            weights = torch.ones(len(X)).to(self.hyperparameters['device'])

        self.model.train()

        dataset = TensorDataset(X, y, weights)
        dataloader = DataLoader(dataset, batch_size=self.hyperparameters['batch_size'])

        for inputs, targets, sample_weights in dataloader:
            self.optimizer.zero_grad()
            outputs = self.model(inputs).squeeze()
            targets = targets.squeeze()
            losses = self.criterion(outputs, targets)
            weighted_loss = (losses * sample_weights).mean()
            weighted_loss.backward()
            self.optimizer.step()

    def get_parameters(self):
        return {name: param.clone() for name, param in self.model.state_dict().items()}

    def set_parameters(self, parameters):
        self.model.load_state_dict(parameters)

    def predict(self, X):
        with torch.no_grad():
            outputs = self.model(X)
            _, predicted = torch.max(outputs.data, 1)
            return predicted

    def predict_proba(self, X, is_tensor=False, is_eval=False):
        if is_eval:
            self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)
        probabilities = torch.sigmoid(outputs)  # Convert logits to probabilities
        return probabilities.squeeze() if is_tensor else probabilities.squeeze().cpu().numpy()
