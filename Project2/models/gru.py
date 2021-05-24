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
        num_units = 2 * tau
        self.path = path
        self.tau = tau
        self.horizon = horizon
        dataset = TimeSeriesDataset(self.path, self.tau, self.horizon)
        self.scale = torch.from_numpy(dataset.get_scale()).T
        self.gru = nn.GRU(1, num_units, batch_first=True)
        self.linear = nn.Linear(num_units, 1)
        self.dropout = nn.Dropout(dropout_prob)
        self.backbone = nn.Sequential(
            self.gru,
            self.dropout,
            self.linear,
        )

    def forward(self, X):
        _, out = self.gru(X.T.unsqueeze(-1), None)
        out = self.dropout(out)
        out = self.linear(out[:, -1, :])
        return out

    def configure_optimizers(self):
        optimizer = optim.Adam([
            {'params': self.backbone.parameters()},
        ], lr=1e-3)
        exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
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
        # loss = F.mse_loss((y_hat * self.test_scale).T, (y * self.test_scale).T)
        rse = get_rse(y_hat.reshape((self.n, -1)).cpu().detach().numpy(), y.reshape((self.n, -1)).cpu().detach().numpy())
        corr = get_corr(y_hat.reshape((self.n, -1)).cpu().detach().numpy(), y.reshape((self.n, -1)).cpu().detach().numpy())

        self.log('test_loss', loss)
        self.log('rse', rse)
        self.log('corr', corr)

        return {'test_loss': loss, 'rse': rse, 'corr': corr}

    def train_dataloader(self):
        # REQUIRED

        train_data = TimeSeriesDataset(self.path, self.tau, self.horizon, train=True)
        train_loader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=0)

        return train_loader

    def val_dataloader(self):

        val_data = TimeSeriesDataset(self.path, self.tau, self.horizon, valid=True)
        val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=0)

        return val_loader

    def test_dataloader(self):
        test_data = TimeSeriesDataset(self.path, self.tau, self.horizon, test=True)
        test_loader = DataLoader(test_data, batch_size=test_data.__len__(), shuffle=False, num_workers=0)

        return test_loader


def grid_search(path: str, params: dict, gpu: int):
    gpus = -1 if gpu != 0 else 0
    tau_list = params['tau']
    horizon = params['horizon']
    dropout = params['dropout']
    best_trainer = None
    best_rmse = 10000000
    for tau in tau_list:
        model = NeuralNetwork(path, tau, horizon, dropout)
        trainer = pl.Trainer(max_epochs=500, callbacks=[EarlyStopping(monitor='val_loss')], gpus=gpus)
        trainer.fit(model)
        rmse = torch.sqrt(trainer.callback_metrics['val_loss'])
        if rmse < best_rmse:
            best_trainer = trainer
            best_rmse = rmse
    best_trainer.test()
    print("Root Mean Squared Error: " + str(torch.sqrt(best_trainer.callback_metrics['test_loss'])))
    print("Root Relative Squared Error: " + str(best_trainer.callback_metrics['rse']))
    print("Empirical Correlation Coefficient: " + str(best_trainer.callback_metrics['corr']))


def train_and_predict(path: str, horizon: int, dropout: float, gpu: int) -> (np.ndarray, np.ndarray):
    params = {'tau': [2 ** 6],
              'horizon': horizon,
              'dropout': dropout}
    grid_search(path, params=params, gpu=gpu)
