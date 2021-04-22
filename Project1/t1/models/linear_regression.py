from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from utils.preprocessing import *


def train_and_predict(path: str):
    X, y = generate_data(path, tau=10)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y)
    num_models = X_train.shape[0]
    # model = LinearRegression(n_jobs=4)
    # model.fit(X_train, y_train)
    # predictions = model.predict(X_valid)
    models = list()
    predictions = list()
    for i in range(num_models):
        model = LinearRegression(n_jobs=4)
        model.fit(X_train[i], y_train[i])
        models.append(model)
        predictions.append(model.predict(X_valid[i]))
    print("Root Mean Squared Error: " + str(mean_squared_error(predictions, y_valid, squared=False)))
