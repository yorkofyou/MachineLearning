import os.path
import numpy as np
import pickle as pkl
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from utils.preprocessing import *
from utils.evaluate import *


def train_model(X: np.ndarray, y: np.ndarray, reg: float) -> BaseEstimator:
    model = SVR(kernel='linear', C=reg)
    model.fit(X, y)
    return model


def grid_search(path: str, params: dict, n_jobs: int):
    data = Dataset(path)
    tau_list = params['tau']
    reg_list = params['reg']
    horizon = params['horizon']
    name = os.path.split(path)[-1].split('.')[0]
    best_model = None
    best_rmse = 10000000
    test_rmse = None
    test_rse = None
    test_corr = None
    for tau in tqdm(tau_list):
        X_train, X_valid, X_test, y_train, y_valid, y_test = data.generate_data(tau=tau, horizon=horizon)
        num_models = X_train.shape[0]
        for reg in reg_list:
            models = Parallel(n_jobs=n_jobs)(delayed(train_model)(X_train[i], y_train[i], reg) for i in range(num_models))
            predictions = np.array([models[i].predict(X_valid[i]) for i in range(num_models)])
            rmse = mean_squared_error((y_valid.T * data.scale).reshape((-1)), (predictions.T * data.scale).reshape((-1)), squared=False)
            if rmse < best_rmse:
                best_model = models
                best_rmse = rmse
                best_predictions = np.array([models[i].predict(X_test[i]) for i in range(num_models)])
                test_rmse = mean_squared_error((y_test.T * data.scale).reshape((-1)),
                                               (best_predictions.T * data.scale).reshape((-1)), squared=False)
                test_rse = get_rse((best_predictions.T * data.scale).reshape((data.n, -1)), (y_test.T * data.scale).reshape((data.n, -1)))
                test_corr = get_corr((best_predictions.T * data.scale).reshape((data.n, -1)), (y_test.T * data.scale).reshape((data.n, -1)))
    filename = os.path.join('models', 'save', '_'.join(['svr', name])) + '.pkl'
    pkl.dump(best_model, open(filename, 'wb'))
    print("Root Mean Squared Error: " + str(test_rmse))
    print("Root Relative Squared Error: " + str(test_rse))
    print("Empirical Correlation Coefficient: " + str(test_corr))


def train_and_predict(path: str, horizon: int, n_jobs: int) -> (np.ndarray, np.ndarray):
    params = {'tau': [2**i for i in range(10)],
              'reg': [2**i for i in range(-10, 11, 2)],
              'horizon': horizon}
    grid_search(path, params=params, n_jobs=n_jobs)
