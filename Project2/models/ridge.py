import os.path
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, RidgeCV
from utils.preprocessing import *
from utils.evaluate import *
from utils.plot import *


def train_model(X: np.ndarray, y: np.ndarray, reg: float) -> BaseEstimator:
    model = Ridge(alpha=reg)
    # model = RidgeCV(alphas=[0.01, 0.1, 0.5, 1, 3, 5, 7, 10, 20, 100, 500, 1000, 2000, 5000, 10000, 20000, 50000])
    model.fit(X, y)
    # print("Best alpha: ", model.alpha_)
    return model


def grid_search(data, params):
    tau_list = params['tau']
    reg_list = params['reg']
    horizon = params['horizon']
    best_rmse = 10000
    best_predictions = None
    y_valid = None
    for tau in tau_list:
        for reg in reg_list:
            X_train, X_valid, X_test, y_train, y_valid, y_test = data.generate_data(tau=tau, horizon=horizon)
            print("Completed.")
            print("Start training...")
            num_models = X_train.shape[0]
            models = Parallel(n_jobs=4)(delayed(train_model)(X_train[i], y_train[i], reg) for i in tqdm(range(num_models)))
            predictions = np.array([models[i].predict(X_valid[i]) for i in range(num_models)])
            rmse = mean_squared_error(y_valid.reshape((-1)), predictions.reshape((-1)), squared=False)
            if rmse < best_rmse:
                best_rmse = rmse
                best_models = models
                best_predictions = predictions
    print("Root Mean Squared Error: " + str(best_rmse))
    print("Root Relative Squared Error: " + str(get_rse(best_predictions, y_valid)))
    print("Empirical Correlation Coefficient: " + str(get_corr(best_predictions, y_valid)))
    return best_predictions


def train_and_predict(path: str, horizon: int) -> (np.ndarray, np.ndarray):
    print("Data loading ", end='')
    data = Dataset(path)
    params = {'tau': [2**i for i in range(10)],
              'reg': [2**i for i in range(-10, 11)],
              'horizon': horizon}
    predictions = grid_search(data, params=params)


train_and_predict('../datasets/traffic.txt', horizon=24)
