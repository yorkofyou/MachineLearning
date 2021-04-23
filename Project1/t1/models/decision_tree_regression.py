import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from utils.preprocessing import *


def train_model(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    parameters = {'criterion': ['mse', 'mae'],
                  'max_depth': [2, 5, 10, 20],
                  'min_samples_split': [2, 3, 5],
                  'min_samples_leaf': [1, 5, 8],
                  'max_features': ['log2', 'sqrt', 'auto']}
    model = DecisionTreeRegressor(random_state=1)
    grid_object = GridSearchCV(model, parameters)
    grid_object.fit(X, y)
    return grid_object


def train_and_predict(path: str, tau: int, normalize=False):
    data = Dataset(path)
    X_train, X_valid, y_train, y_valid = data.generate_data(tau=tau)
    num_models = X_train.shape[0]
    models = Parallel(n_jobs=4)(delayed(train_model)(X_train[i], y_train[i]) for i in range(num_models))
    if normalize:
        predictions = np.array([models[i].predict(X_valid[i]) for i in range(num_models)]) * data.valid_std + data.valid_mean
    else:
        predictions = np.array([models[i].predict(X_valid[i]) for i in range(num_models)])
    print("Root Mean Squared Error: " + str(mean_squared_error(predictions, y_valid, squared=False)))


train_and_predict('../traffic.txt', tau=2)
