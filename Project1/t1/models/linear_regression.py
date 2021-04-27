import os.path
import pickle
from random import seed, randint

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from utils.preprocessing import *
from utils.plot import *


def train_and_predict(path: str, tau: int, normalize=False) -> (np.ndarray, np.ndarray):
    seed(1)
    name = os.path.split(path)[-1].split('.')[0]
    data = Dataset(path)
    X_train, X_valid, y_train, y_valid = data.generate_data(tau=tau, time='1', normalize=normalize)
    model = LinearRegression()
    model.fit(X_train, y_train)
    filename = os.path.join('models', 'save', '_'.join(['linear_regression', name])) + '.sav'
    pickle.dump(model, open(filename, 'wb'))
    if normalize:
        predictions = model.predict(X_valid)
        predictions = data.scaler.inverse_transform(predictions)
    else:
        predictions = model.predict(X_valid)
    print("Root Mean Squared Error: " + str(mean_squared_error(predictions, y_valid, squared=False)))
    predictions = data.reshape_labels(predictions)
    y_valid = data.reshape_labels(y_valid)
    # value = randint(0, data.n)
    # plot_results(name, predictions[value], y_valid[value])
    return predictions, y_valid


# train_and_predict('../commodity.txt', tau=7)
