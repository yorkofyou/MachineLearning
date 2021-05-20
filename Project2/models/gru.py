import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from torch import nn, optim
from torch.utils.data import DataLoader
from utils.dataset import *
from utils.evaluate import *
from utils.plot import *


class NeuralNetwork(nn.Module):
    def __init__(self, tau: int, dropout_prob: float):
        super(NeuralNetwork, self).__init__()
        num_units = 512
        self.gru = nn.GRU(1, num_units, batch_first=True)
        self.linear = nn.Linear(num_units, 1)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, X):
        _, out = self.gru(X.unsqueeze(-1), None)
        out = self.dropout(out)
        out = self.linear(torch.squeeze(out[-1:, :, :], 0))
        return out.squeeze()


def train(model, loss_fn, optimizer, training_loader, validation_loader, epochs):
    epoch_number = 0
    best_vloss = 1_000_000
    for epoch in tqdm(range(epochs)):
        # print('EPOCH {}:'.format(epoch_number + 1))
        model.train(True)
        running_loss = 0.
        last_loss = 0.

        for i, data in enumerate(training_loader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 5 == 4:
                last_loss = running_loss / 5 # Loss per batch
                # print(' batch {} loss: {}'.format(i + 1, last_loss))
                running_loss = 0.

        avg_loss = last_loss
        model.train(False)
        running_vloss = 0.0
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss
        avg_vloss = running_vloss / (i + 1)
        # print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss

        epoch_number += 1
    return model


def predict(model, loss_fn, testing_loader):
    model.eval()
    running_loss = 0.
    prediction = np.array([])
    y_valid = np.array([])
    with torch.no_grad():
        for i, data in enumerate(testing_loader):
            inputs, labels = data
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            running_loss += loss.item()
            if i % 5 == 4:
                last_loss = running_loss / 5 # Loss per batch
                # print(' batch {} loss: {}'.format(i + 1, last_loss))
            y_valid = np.concatenate((y_valid, labels.numpy()), axis=None) if y_valid.size else labels.numpy()
            prediction = np.concatenate((prediction, outputs.detach().numpy())) if prediction.size else outputs.detach().numpy()
    return y_valid, prediction


def train_and_predict(path: str, tau: int, horizon: int, dropout: float, gpu: int) -> (np.ndarray, np.ndarray):
    print("Data loading ", end='')
    training_data = TimeSeriesDataset(path, tau, horizon, train=True)
    validation_data = TimeSeriesDataset(path, tau, horizon, valid=True)
    testing_data = TimeSeriesDataset(path, tau, horizon, test=True)
    print("Completed.")
    print("Start training...")
    training_loader = DataLoader(training_data, batch_size=4, shuffle=True, num_workers=0)
    validation_loader = DataLoader(validation_data, batch_size=4, shuffle=False, num_workers=0)
    testing_loader = DataLoader(testing_data, batch_size=4, shuffle=False, num_workers=0)
    model = NeuralNetwork(tau=tau, dropout_prob=dropout)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model = train(model, loss_fn=loss_fn, optimizer=optimizer, training_loader=training_loader, validation_loader=validation_loader, epochs=100)
    y_valid, predictions = predict(model, loss_fn=loss_fn, testing_loader=testing_loader)
    print("Root Mean Squared Error: " + str(mean_squared_error(y_valid.reshape((-1)), predictions.reshape((-1)), squared=False)))
    # print("Root Relative Squared Error: " + str(get_rse(predictions, y_valid)))
    # print("Empirical Correlation Coefficient: " + str(get_corr(predictions, y_valid)))
