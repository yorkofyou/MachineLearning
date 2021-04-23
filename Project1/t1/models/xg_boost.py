import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from utils.preprocessing import *
from utils.plot import *


def train_model(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    parameters = {'n_estimators': [100, 200, 300, 500, 1000],
                  'eval_metric': ['rmse', 'mae'],
                  'max_depth': [2, 5, 10, 20]}
    model = XGBRegressor(random_state=1)
    grid_object = GridSearchCV(model, parameters)
    grid_object.fit(X, y)
    return grid_object


def train_and_predict(path: str, normalize=False):
    data = Dataset(path)
    X_train, X_valid, y_train, y_valid = data.generate_data(tau=7, normalize=normalize)
    num_models = X_train.shape[0]
    models = Parallel(n_jobs=4)(delayed(train_model)(X_train[i], y_train[i]) for i in range(num_models))
    if normalize:
        predictions = np.array([models[i].predict(X_valid[i]) for i in range(num_models)]) * data.valid_std + data.valid_mean
    else:
        predictions = np.array([models[i].predict(X_valid[i]) for i in range(num_models)])
    print("Root Mean Squared Error: " + str(mean_squared_error(predictions, y_valid, squared=False)))
    # plot_results(predictions, y_valid)


# train_and_predict('../commodity.txt')
