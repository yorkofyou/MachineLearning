import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    def __init__(self, path, tau, horizon, train=True, train_size=0.8):
        data = pd.read_csv(path, header=None, delimiter=',').values
        data = data / np.max(np.abs(data), axis=0, keepdims=True)
        self.t, self.n = data.shape
        assert horizon > 0
        train_index = int(train_size*self.t)
        features = np.array([data[i-tau: i] for i in range(tau, self.t-horizon+1)])
        labels = data[tau+horizon-1:]
        features = np.transpose(features, (2, 0, 1))
        labels = np.transpose(labels, (1, 0))
        if train:
            features = features[:, :train_index, :]
            labels = labels[:, :train_index]
        else:
            features = features[:, train_index:, :]
            labels = labels[:, train_index:]
        self.features = features
        self.labels = labels
        self.model_id = 0

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.features[self.model_id, idx])).float(), torch.from_numpy(np.array(self.labels[self.model_id, idx])).float()

    def __len__(self):
        return self.features.shape[1]

    def get_num_models(self):
        return self.n

    def set_id(self, model_id):
        self.model_id = model_id
