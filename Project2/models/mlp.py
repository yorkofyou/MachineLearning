import os.path
import numpy as np
import torch
import pytorch_lightning as pl
from tqdm import tqdm
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
        num_units = 512
        self.path = path
        self.tau = tau
        self.horizon = horizon
        self.ll1 = nn.Linear(tau+2, num_units)
        self.ll2 = nn.Linear(num_units, 1)
        self.dropout = nn.Dropout(dropout_prob)
        self.backbone = nn.Sequential(self.ll1, self.ll2)

    def forward(self, X):
        X = self.backbone(X)
        return X.squeeze()

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

        loss = F.mse_loss(y_hat, y)
        return {'test_loss': loss}

    def train_dataloader(self):
        # REQUIRED

        train_data = TimeSeriesDataset(self.path, self.tau, self.horizon, train=True)
        train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=0)

        return train_loader

    def val_dataloader(self):

        val_data = TimeSeriesDataset(self.path, self.tau, self.horizon, valid=True)
        val_loader = DataLoader(val_data, batch_size=64, shuffle=False, num_workers=0)

        return val_loader

    def test_dataloader(self):
        test_data = TimeSeriesDataset(self.path, self.tau, self.horizon, test=True)
        test_loader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=0)

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


def train_and_predict(path: str, horizon: int, dropout: float, gpu: int) -> (np.ndarray, np.ndarray):
    params = {'tau': [2**i for i in range(10)],
              'horizon': horizon,
              'dropout': dropout}
    grid_search(path, params=params, gpu=gpu)
