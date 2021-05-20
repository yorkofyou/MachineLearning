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
        num_units = tau
        self.ll1 = nn.Linear(num_units, num_units)
        self.ll2 = nn.Linear(num_units, 1)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, X):
        out = self.dropout(self.ll1(X))
        out = self.dropout(self.ll2(out))
        return out.squeeze()


def train(model, loss_fn, optimizer, training_loader, epochs):
    epoch_number = 0
    for epoch in range(epochs):
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
            if i % 1000 == 999:
                last_loss = running_loss / 1000 # Loss per batch
                print(' batch {} loss: {}'.format(i + 1, last_loss))
                running_loss = 0.

        avg_loss = last_loss
        model.train(False)
        running_vloss = 0.0
        # for i, vdata in enumerate(validation_loader):
        #     vinputs, vlabels = vdata
        #     voutputs = model(vinputs)
        #     vloss = loss_fn(voutputs, vlabels)
        #     running_vloss += vloss
        #
        # avg_vloss = running_vloss / (i + 1)
        # print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
        # if avg_vloss < best_vloss:
        #     best_vloss = avg_vloss

        epoch_number += 1
    return model


def predict(model, loss_fn, validation_loader):
    model.eval()
    running_loss = 0.
    prediction = np.array([])
    y_valid = np.array([])
    with torch.no_grad():
        for i, data in enumerate(validation_loader):
            inputs, labels = data
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            running_loss += loss.item()
            if i % 1000 == 999:
                last_loss = running_loss / 1000 # Loss per batch
                print(' batch {} loss: {}'.format(i + 1, last_loss))
            y_valid = np.concatenate((y_valid, labels.numpy()), axis=None) if y_valid.size else labels.numpy()
            prediction = np.concatenate((prediction, outputs.detach().numpy())) if prediction.size else outputs.detach().numpy()
    return y_valid, prediction


def train_and_predict(path: str, tau: int, horizon: int) -> (np.ndarray, np.ndarray):
    print("Data loading ", end='')
    # data = Dataset(path)
    # X_train, X_valid, y_train, y_valid = data.generate_data(tau=tau, horizon=horizon)
    training_data = TimeSeriesDataset(path, tau, horizon, train=True)
    validation_data = TimeSeriesDataset(path, tau, horizon, train=False)
    print("Completed.")
    print("Start training...")
    # num_models = X_train.shape[0]
    num_models = training_data.get_num_models()
    models = list()
    y_valid = list()
    predictions = list()
    for i in tqdm(range(num_models)):
        training_data.set_id(i)
        validation_data.set_id(i)
        training_loader = DataLoader(training_data, batch_size=32, shuffle=False, num_workers=0)
        validation_loader = DataLoader(validation_data, batch_size=32, shuffle=False, num_workers=0)
        model = NeuralNetwork(tau=tau, dropout_prob=0.2)
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        model = train(model, loss_fn=loss_fn, optimizer=optimizer, training_loader=training_loader, epochs=500)
        label, prediction = predict(model, loss_fn=loss_fn, validation_loader=validation_loader)
        models.append(model)
        y_valid.append(label)
        predictions.append(prediction)
    y_valid = np.array(y_valid)
    predictions = np.array(predictions)
    print("Root Mean Squared Error: " + str(mean_squared_error(y_valid.reshape((-1)), predictions.reshape((-1)), squared=False)))
    print("Root Relative Squared Error: " + str(get_rse(predictions, y_valid)))
    print("Empirical Correlation Coefficient: " + str(get_corr(predictions, y_valid)))
    # predictions = np.array([models[i].predict(X_valid[i]) for i in range(num_models)])
    # print("Root Mean Squared Error: " + str(mean_squared_error(predictions, y_valid, squared=False)))
    # predictions = data.reshape_labels(predictions)
    # y_valid = data.reshape_labels(y_valid)
    # return predictions, y_valid


train_and_predict('../datasets/commodity.txt', tau=2, horizon=3)
