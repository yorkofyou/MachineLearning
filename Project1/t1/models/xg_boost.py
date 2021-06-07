import os.path
import pickle
from random import seed

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from utils.preprocessing import *
from utils.plot import *


def train_and_predict(path: str, tau: int, normalize=False) -> (np.ndarray, np.ndarray):
    seed(1)
    name = os.path.split(path)[-1].split('.')[0]
    data = Dataset(path)
    X_train, X_valid, y_train, y_valid = data.generate_data(tau=tau, time='1', normalize=normalize)
    if name != 'commodity':
        n_estimators = [100, 200, 300, 500, 1000]
    else:
        n_estimators = [10, 20, 50, 100, 200]
    parameters = {'n_estimators': n_estimators,
                  'eval_metric': ['rmse', 'mae'],
                  'max_depth': [2, 5, 10],
                  'min_child_weight': [1, 3, 5]}
    model = XGBRegressor(random_state=1)
    grid_object = GridSearchCV(model, parameters)
    grid_object.fit(X_train, y_train)
    filename = os.path.join('models', 'save', '_'.join(['xg_boost', name])) + '.pkl'
    pickle.dump(grid_object, open(filename, 'wb'))
    print(grid_object.best_params_)
    if normalize:
        predictions = grid_object.predict(X_valid)
        predictions = data.scaler.inverse_transform(predictions)
    else:
        predictions = grid_object.predict(X_valid)
    print("Root Mean Squared Error: " + str(mean_squared_error(predictions, y_valid, squared=False)))
    predictions = data.reshape_labels(predictions)
    y_valid = data.reshape_labels(y_valid)
    return predictions, y_valid
