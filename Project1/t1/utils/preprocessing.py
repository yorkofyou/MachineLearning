import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class Dataset:
    def __init__(self, path: str, train_size=0.8):
        self.train_size = train_size
        self.data = pd.read_csv(path, header=None, delimiter=',').values
        t, _ = self.data.shape
        index = int(t * train_size)
        self.train_mean, self.train_std = self.data[:index].mean(axis=0, keepdims=True).T, self.data[:index].std(axis=0, keepdims=True).T
        self.valid_mean, self.valid_std = self.data[index:].mean(axis=0, keepdims=True).T, self.data[index:].std(axis=0, keepdims=True).T
        self.data = np.delete(self.data, np.argwhere(self.valid_std == 0), axis=1)
        self.train_mean = np.delete(self.train_mean, np.argwhere(self.valid_std == 0), axis=0)
        self.train_std = np.delete(self.train_std, np.argwhere(self.valid_std == 0), axis=0)
        self.valid_mean = np.delete(self.valid_mean, np.argwhere(self.valid_std == 0), axis=0)
        self.valid_std = np.delete(self.valid_std, np.argwhere(self.valid_std == 0), axis=0)

    def generate_data(self, tau: int, normalize=False) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        t, _ = self.data.shape
        features = [np.vstack((self.data[i-tau: i], np.amin(self.data[: i], axis=0), np.amax(self.data[: i], axis=0))) for i in range(tau, t)]
        labels = self.data[tau:]
        return self.train_test_split(np.transpose(features, (2, 0, 1)), np.transpose(labels, (1, 0)), normalize=normalize)  # (N, T, F) and (N, T, )

    def train_test_split(self, X: np.ndarray, y: np.ndarray, train_size=0.8, normalize=False) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        n, t, f = X.shape
        index = int(t * train_size)
        X_train, y_train = X[:, :index, :], y[:, :index]
        X_valid, y_valid = X[:, index:, :], y[:, index:]
        if normalize:
            X_train = ((X_train.reshape((n, -1)) - self.train_mean) / self.train_std).reshape((n, index, f))
            y_train = (y_train - self.train_mean) / self.train_std
            X_valid = ((X_valid.reshape((n, -1)) - self.valid_mean) / self.valid_std).reshape((n, t-index, f))
        # X_train, X_valid = X_train.reshape((-1, f)), X_valid.reshape((-1, f))
        # y_train, y_valid = y_train.reshape((-1)), y_valid.reshape((-1))
        return X_train, X_valid, y_train, y_valid

    def feature_extraction(self):
        sns.lineplot(data=self.data[1])
        plt.show()
