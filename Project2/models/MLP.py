import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from skorch import NeuralNetRegressor
from torch import nn, optim
from torch.functional import F
from utils.preprocessing import *
from utils.plot import *


class NeuralNetwork(nn.Module):
    def __init__(self, tau):
        super(NeuralNetwork, self).__init__()
        num_units = tau+2
        self.ll1 = nn.Linear(num_units, 2*num_units)
        self.ll2 = nn.Linear(2*num_units, 4*num_units)
        self.ll3 = nn.Linear(4*num_units, 2*num_units)
        self.ll4 = nn.Linear(2*num_units, num_units)
        self.ll5 = nn.Linear(num_units, 1)

    def forward(self, X):
        X = F.relu(self.ll1(X))
        X = F.relu(self.ll2(X))
        X = F.relu(self.ll3(X))
        X = F.relu(self.ll4(X))
        X = F.relu(self.ll5(X))
        return X


def train_model(X, y, tau: int) -> NeuralNetRegressor:
    net = NeuralNetwork(tau=tau)
    model = NeuralNetRegressor(
        net,
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
