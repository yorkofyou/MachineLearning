import pickle
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from utils.preprocessing import *
from utils.plot import *


def train_and_predict(path: str):
    X, y = generate_data(path, tau=7)
    N = X.shape[0]
    X_train, X_valid, y_train, y_valid = train_test_split(X, y)
    models = list()
    predictions = list()
    for i in range(N):
        model = XGBRegressor(random_state=1, n_estimators=500)
        model.fit(X_train[i], y_train[i],
              verbose=False)
        models.append(model)
        predictions.append(model.predict(X_valid[i]))
    # predictions = predictions.reshape((N, -1))
    # y_valid = y_valid.reshape((N, -1))
    print("Root Mean Squared Error: " + str(mean_squared_error(predictions, y_valid, squared=False)))
    plot_results(predictions, y_valid)


train_and_predict('../commodity.txt')
