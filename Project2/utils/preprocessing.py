import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


class Dataset:
    def __init__(self, path: str):
        self.data = pd.read_csv(path, header=None, delimiter=',').values
        self.t, self.n = self.data.shape

    def generate_data(self, tau: int, horizon: int) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        # self.data = self.data / np.max(np.abs(self.data), axis=0, keepdims=True)
        assert horizon > 0
        # features = np.array([self.data[i-tau: i] for i in range(tau, self.t-horizon+1)])
        features = np.array([np.vstack((self.data[i-tau: i], np.amin(self.data[: i], axis=0), np.amax(self.data[: i], axis=0))) for i in range(tau, self.t-horizon+1)])
        labels = self.data[tau+horizon-1:]
        return self.train_test_split(np.transpose(features, (2, 0, 1)), np.transpose(labels, (1, 0)))  # (N, T, F) and (N, T, )

    def train_test_split(self, X: np.ndarray, y: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        n, t, f = X.shape
        train_index = int(t * 0.6)
        valid_index = int(t * 0.8)
        X_train, y_train = X[:, :train_index, :], y[:, :train_index]
        X_valid, y_valid = X[:, train_index: valid_index], y[:, train_index: valid_index]
        X_test, y_test = X[:,  valid_index:, :], y[:, valid_index:]
        # X_train, X_valid = X_train.reshape((-1, f)), X_valid.reshape((-1, f))
        # y_train, y_valid = y_train.reshape((-1)), y_valid.reshape((-1))
        # Shuffle train set
        rand_index = np.random.permutation(train_index)
        X_train = X_train[:, rand_index, :]
        y_train = y_train[:, rand_index]
        return X_train, X_valid, X_test, y_train, y_valid, y_test

    def reshape_labels(self, labels: np.ndarray) -> np.ndarray:
        return labels.reshape((self.n, -1))

    def feature_extraction(self):
        sns.lineplot(data=self.data[1])
        plt.show()
