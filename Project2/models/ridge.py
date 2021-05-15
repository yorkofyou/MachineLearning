import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, RidgeCV
from utils.preprocessing import *
from utils.plot import *


def train_model(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    model = Ridge()
    model.fit(X, y)
    return model


def train_and_predict(path: str, tau: int, horizon: int, train_size: float, normalize=False) -> (np.ndarray, np.ndarray):
    print("Data loading ", end='')
    data = Dataset(path)
    X_train, X_valid, y_train, y_valid = data.generate_data(tau=tau, horizon=horizon, train_size=train_size, normalize=normalize)
    print("Completed.")
    print("Start training...")
    num_models = X_train.shape[0]
    models = Parallel(n_jobs=4)(delayed(train_model)(X_train[i], y_train[i]) for i in tqdm(range(num_models)))
    if normalize:
        predictions = np.array([models[i].predict(X_valid[i]) for i in range(num_models)])
        predictions = data.scaler.inverse_transform(predictions)
    else:
        predictions = np.array([models[i].predict(X_valid[i]) for i in range(num_models)])
    print("Root Mean Squared Error: " + str(mean_squared_error(predictions, y_valid, squared=False)))
    predictions = data.reshape_labels(predictions)
    y_valid = data.reshape_labels(y_valid)
    return predictions, y_valid


train_and_predict('../datasets/commodity.txt', tau=7, horizon=1, train_size=0.8)
