import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    def __init__(self, path, tau, horizon):
        data = pd.read_csv(path, header=None, delimiter=',').values
        self.t, self.n = data.shape
        assert horizon > 0
        features = np.array([np.vstack((data[i-tau: i], np.amin(data[: i], axis=0), np.amax(data[: i], axis=0))) for i in range(tau, self.t)])
        labels = data[tau+horizon-1:]
        self.features = np.transpose(features, (2, 0, 1))
        self.labels = np.transpose(labels, (1, 0))

    def __getitem__(self, idx):
        return torch.from_numpy(self.features[idx]).float(), torch.from_numpy(self.labels[idx]).float().reshape((-1, 1))

    def __len__(self):
        return self.features.shape[1]

    def get_num_models(self):
        return self.n
