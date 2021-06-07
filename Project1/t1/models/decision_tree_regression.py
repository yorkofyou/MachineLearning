import os.path
import pickle
from random import seed

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from utils.preprocessing import *
from utils.plot import *


def train_and_predict(path: str, tau: int, normalize=False) -> (np.ndarray, np.ndarray):
    seed(1)
    name = os.path.split(path)[-1].split('.')[0]
    data = Dataset(path)
    X_train, X_valid, y_train, y_valid = data.generate_data(tau=tau, time='1', normalize=normalize)
    parameters = {'criterion': ['mse', 'mae'],
                  'max_depth': [2, 5, 10, 20],
                  'min_samples_split': [2, 3, 5],
                  'min_samples_leaf': [1, 5, 8],
                  'max_features': ['log2', 'sqrt', 'auto']}
    model = DecisionTreeRegressor(random_state=1)
    grid_object = GridSearchCV(model, parameters)
    grid_object.fit(X_train, y_train)
    filename = os.path.join('models', 'save', '_'.join(['decision_tree_regression', name])) + '.pkl'
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
