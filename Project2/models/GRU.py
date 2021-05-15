import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from skorch import NeuralNet
from torch import nn, optim
from torch.functional import F
from utils.preprocessing import *
from utils.plot import *


class NeuralNetwork(nn.Module):
    def __init__(self, tau):
        super(NeuralNetwork, self).__init__()
        num_units = tau+2
        self.gru = nn.GRU(1, num_units)
        self.linear = nn.Linear(num_units, 1)

    def forward(self, X):
        out, _ = self.gru(X.T.unsqueeze(-1), None)
        out = self.linear(out[-1])
        return out


def train_model(X, y, tau: int) -> NeuralNet:
    net = NeuralNetwork(tau=tau)
    model = NeuralNet(
        net,
        criterion=nn.MSELoss,
        batch_size=32,
        max_epochs=500,
        lr=0.1,
        optimizer=optim.Adam
    )
    model.fit(X, y)
    return model


def train_and_predict(path: str, tau: int, horizon: int, train_size: float, normalize=False) -> (np.ndarray, np.ndarray):
    print("Data loading ", end='')
    data = Dataset(path)
    X_train, X_valid, y_train, y_valid = data.generate_data(tau=tau, horizon=horizon, train_size=train_size, normalize=normalize)
    print("Completed.")
    print("Start training...")
    num_models = X_train.shape[0]
    models = list()
    for i in tqdm(range(num_models)):
        model = train_model(torch.from_numpy(X_train[i]).float(), torch.from_numpy(y_train[i]).float().reshape((-1, 1)), tau)
        models.append(model)
    if normalize:
        predictions = np.array([models[i].predict(X_valid[i]) for i in range(num_models)])
        predictions = data.scaler.inverse_transform(predictions)
    else:
        predictions = np.array([models[i].predict(X_valid[i]) for i in range(num_models)])
    print("Root Mean Squared Error: " + str(mean_squared_error(predictions, y_valid, squared=False)))
    predictions = data.reshape_labels(predictions)
    y_valid = data.reshape_labels(y_valid)
    return predictions, y_valid


train_and_predict('../datasets/commodity.txt', tau=7, horizon=1, train_size=0.8)
