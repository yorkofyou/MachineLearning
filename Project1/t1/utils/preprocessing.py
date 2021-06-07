import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


class Dataset:
    def __init__(self, path: str, train_size=0.8):
        self.train_size = train_size
        self.data = pd.read_csv(path, header=None, delimiter=',').values
        t, self.n = self.data.shape
        index = int(t * train_size)
        self.scaler = StandardScaler().fit(self.data[:index])
        self.train_mean, self.train_std = self.data[:index].mean(axis=0, keepdims=True).T, self.data[:index].std(axis=0, keepdims=True).T
        self.valid_mean, self.valid_std = self.data[index:].mean(axis=0, keepdims=True).T, self.data[index:].std(axis=0, keepdims=True).T

    def generate_data(self, tau: int, time: str, normalize=False) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        t, _ = self.data.shape
        features = np.array([np.vstack((self.data[i-tau: i], np.amin(self.data[: i], axis=0), np.amax(self.data[: i], axis=0))) for i in range(tau, t)])
        if time == '1':
            labels = self.data[tau:]
        elif time == '3':
            labels = np.array([self.data[i: i+3] for i in range(tau, t-3)])
        elif time == '6':
            labels = np.array([self.data[i: i+6] for i in range(tau, t-6)])
        elif time == '9':
            labels = np.array([self.data[i: i+9] for i in range(tau, t-9)])
        elif time == '12':
            labels = np.array([self.data[i: i+12] for i in range(tau, t-12)])
        else:
            raise ValueError
        return self.train_test_split(np.transpose(features, (2, 0, 1)), np.transpose(labels, (1, 0)), normalize=normalize)  # (N, T, F) and (N, T, )

    def train_test_split(self, X: np.ndarray, y: np.ndarray, train_size=0.8, normalize=False) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        n, t, f = X.shape
        index = int(t * train_size)
        X_train, y_train = X[:, :index, :], y[:, :index]
        X_valid, y_valid = X[:, index:, :], y[:, index:]
        if normalize:
            X_train = self.scaler.transform(X_train.reshape((n, -1))).reshape((n, index, f))
            y_train = self.scaler.transform(y_train)
            X_valid = self.scaler.transform(X_valid.reshape((n, t-index, f))).reshape((n, t-index, f))
        X_train, X_valid = X_train.reshape((-1, f)), X_valid.reshape((-1, f))
        y_train, y_valid = y_train.reshape((-1)), y_valid.reshape((-1))
        return X_train, X_valid, y_train, y_valid

    def reshape_labels(self, labels: np.ndarray) -> np.ndarray:
        return labels.reshape((self.n, -1))

    def feature_extraction(self):
        sns.lineplot(data=self.data[1])
        plt.show()
