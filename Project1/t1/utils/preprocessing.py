import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def feature_extraction(path: str):
    data = np.loadtxt(path, delimiter=',')
    sns.lineplot(data=data[1])
    plt.show()


def generate_data(path: str, tau: int) -> (np.ndarray, np.ndarray):
    data = np.loadtxt(path, delimiter=',')
    t, _ = data.shape
    features = [np.vstack((data[i-tau: i], np.amin(data[: i], axis=0), np.amax(data[: i], axis=0))) for i in range(tau, t)]
    labels = data[tau:]
    return np.transpose(features, (2, 0, 1)), np.transpose(labels, (1, 0))  # (N, T, F) and (N, T, )


def train_test_split(X: np.ndarray, y: np.ndarray, train_size=0.8) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    _, t, f = X.shape
    index = int(t * train_size)
    X_train, y_train = X[:, :index, :], y[:, :index]
    X_valid, y_valid = X[:, index:, :], y[:, index:]
    # X_train, X_valid = X_train.reshape((-1, f)), X_valid.reshape((-1, f))
    # y_train, y_valid = y_train.reshape((-1)), y_valid.reshape((-1))
    return X_train, X_valid, y_train, y_valid
