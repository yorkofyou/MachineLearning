import numpy as np
import torch
import pytorch_lightning as pl
from torch import nn, optim
from torch.functional import F
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from utils.dataset import *
from utils.evaluate import *
from utils.plot import *


class NeuralNetwork(pl.LightningModule):
    def __init__(self, path: str, tau: int, horizon: int, dropout_prob: float):
        super(NeuralNetwork, self).__init__()
        self.path = path
        self.tau = tau
        num_units = tau
        self.horizon = horizon
        dataset = TimeSeriesDataset(self.path, self.tau, self.horizon)
        self.scale = torch.from_numpy(dataset.get_scale()).T
        self.n = dataset.n
        self.gru = nn.GRU(1, num_units, 3, batch_first=True)
        self.linear = nn.Linear(num_units, 1)
        self.dropout = nn.Dropout(dropout_prob)
        self.backbone = nn.Sequential(
            self.gru,
            self.dropout,
            self.linear,
        )

    def forward(self, X):
        _, out = self.gru(X.unsqueeze(-1), None)
        out = self.dropout(out[1])
        out = self.linear(out.squeeze())
        return out

    def configure_optimizers(self):
        optimizer = optim.Adam([
            {'params': self.backbone.parameters()},
        ], lr=1e-3)
        exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [exp_lr_scheduler]

    def training_step(self, batch, batch_idx):
        # REQUIRED
        x, y = batch
        y_hat = self.forward(x)

        loss = F.mse_loss(y_hat, y)

        self.log('train_loss', loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)

        loss = F.mse_loss(y_hat, y)

        self.log('val_loss', loss)
        return {'val_loss': loss}

    def test_step(self, batch, batch_idx):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)

        loss = F.mse_loss((y_hat.reshape((self.n, -1)) * self.scale).reshape((-1)), (y.reshape((self.n, -1)) * self.scale).reshape((-1)))
        rse = get_rse((y_hat.reshape((self.n, -1)) * self.scale).cpu().detach().numpy(), (y.reshape((self.n, -1)) * self.scale).cpu().detach().numpy())
        corr = get_corr((y_hat.reshape((self.n, -1)) * self.scale).cpu().detach().numpy(), (y.reshape((self.n, -1)) * self.scale).cpu().detach().numpy())

        self.log('test_loss', loss)
        self.log('rse', rse)
        self.log('corr', corr)

        return {'test_loss': loss, 'rse': rse, 'corr': corr}

    def train_dataloader(self):
        # REQUIRED

        train_data = TimeSeriesDataset(self.path, self.tau, self.horizon, train=True)
        train_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=0)

        return train_loader

    def val_dataloader(self):

        val_data = TimeSeriesDataset(self.path, self.tau, self.horizon, valid=True)
        val_loader = DataLoader(val_data, batch_size=128, shuffle=False, num_workers=0)

        return val_loader

    def test_dataloader(self):
        test_data = TimeSeriesDataset(self.path, self.tau, self.horizon, test=True)
        test_loader = DataLoader(test_data, batch_size=128, shuffle=False, num_workers=0)

        return test_loader

    def get_test_labels(self):
        test_data = TimeSeriesDataset(self.path, self.tau, self.horizon, test=True)
        return test_data.labels


def test(model: pl.LightningModule):
    dataloader = model.test_dataloader()
    predictions = list()
    labels = list()
    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            prediction = model.forward(x)
            predictions.append(prediction)
            labels.append(y)
    predictions = torch.cat(predictions)
    labels = torch.cat(labels)
    test_rmse = F.mse_loss((predictions.reshape((model.n, -1)) * model.scale).reshape((-1)),
                           (labels.reshape((model.n, -1)) * model.scale).reshape((-1)))
    test_rse = get_rse((predictions.reshape((model.n, -1)) * model.scale).cpu().detach().numpy(),
                       (labels.reshape((model.n, -1)) * model.scale).cpu().detach().numpy())
    test_corr = get_corr((predictions.reshape((model.n, -1)) * model.scale).cpu().detach().numpy(),
                         (labels.reshape((model.n, -1)) * model.scale).cpu().detach().numpy())
    print("Root Mean Squared Error: " + str(torch.sqrt(test_rmse)))
    print("Root Relative Squared Error: " + str(test_rse))
    print("Empirical Correlation Coefficient: " + str(test_corr))


def grid_search(path: str, params: dict, gpu: int):
    gpus = -1 if gpu != 0 else 0
    tau_list = params['tau']
    horizon = params['horizon']
    dropout = params['dropout']
    best_model = None
    best_rmse = 10000000
    for tau in tau_list:
        model = NeuralNetwork(path, tau, horizon, dropout)
        trainer = pl.Trainer(max_epochs=50000, callbacks=[EarlyStopping(monitor='val_loss')], gpus=gpus)
        lr_finder = trainer.tuner.lr_find(model)
        new_lr = lr_finder.suggestion()
        model.hparams.lr = new_lr
        trainer.fit(model)
        rmse = torch.sqrt(trainer.callback_metrics['val_loss'])
        if rmse < best_rmse:
            best_model = model
            best_rmse = rmse
    test(best_model)


def train_and_predict(path: str, horizon: int, dropout: float, gpu: int) -> (np.ndarray, np.ndarray):
    params = {'tau': [2 ** i for i in range(10)],
              'horizon': horizon,
              'dropout': dropout}
    grid_search(path, params=params, gpu=gpu)
