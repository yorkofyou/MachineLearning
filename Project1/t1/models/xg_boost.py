import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from utils.preprocessing import *
from utils.plot import *


def train_model(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    parameters = {'n_estimators': [20, 50, 100, 200, 500],
                  'eval_metric': ['rmse', 'mae'],
                  'max_depth': [2, 5, 10],
                  'min_child_weight': [1, 3, 5]}
    model = XGBRegressor(random_state=1)
    grid_object = GridSearchCV(model, parameters)
    grid_object.fit(X, y)
    print(grid_object.best_params_)
    return grid_object


def train_and_predict(path: str, tau: int, normalize=False):
    data = Dataset(path)
    X_train, X_valid, y_train, y_valid = data.generate_data(tau=tau, normalize=normalize)
    num_models = X_train.shape[0]
    models = Parallel(n_jobs=4)(delayed(train_model)(X_train[i], y_train[i]) for i in range(num_models))
    if normalize:
        predictions = np.array([models[i].predict(X_valid[i]) for i in range(num_models)]) * data.valid_std + data.valid_mean
    else:
        predictions = np.array([models[i].predict(X_valid[i]) for i in range(num_models)])
    print("Root Mean Squared Error: " + str(mean_squared_error(predictions, y_valid, squared=False)))
    # plot_results(predictions, y_valid)


train_and_predict('../electricity.txt', tau=24)
