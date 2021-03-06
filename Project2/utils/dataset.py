import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    def __init__(self, path, tau, horizon, train=False, valid=False, test=False):
        data = pd.read_csv(path, header=None, delimiter=',').values
        self.scale = np.max(np.abs(data), axis=0, keepdims=True)
        data = data / np.max(np.abs(data), axis=0, keepdims=True)
        self.t, self.n = data.shape
        assert horizon > 0
        train_index = int(self.t * 0.6)
        valid_index = int(self.t * 0.8)
        # features = np.array([np.vstack((data[i-tau: i], np.amin(data[: i], axis=0), np.amax(data[: i], axis=0))) for i in range(tau, self.t-horizon+1)])
        features = np.array([data[i-tau: i] for i in range(tau, self.t-horizon+1)])
        labels = data[tau+horizon-1:]
        features = np.transpose(features, (2, 0, 1))
        labels = np.transpose(labels, (1, 0))
        if train:
            features = features[:, :train_index, :]
            labels = labels[:, :train_index]
        elif valid:
            features = features[:, train_index: valid_index, :]
            labels = labels[:, train_index: valid_index]
        elif test:
            features = features[:, valid_index:, :]
            labels = labels[:, valid_index:]
        self.features = features.reshape((-1, features.shape[2]))
        self.labels = labels.reshape((-1, 1))

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.features[idx])).float(), torch.from_numpy(np.array(self.labels[idx])).float()

    def __len__(self):
        return self.features.shape[0]

    def get_scale(self):
        return self.scale
