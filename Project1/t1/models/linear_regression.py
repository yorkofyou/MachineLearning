from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from utils.preprocessing import *


def train_and_predict(path: str, tau: int, normalize=False):
    data = Dataset(path)
    X_train, X_valid, y_train, y_valid = data.generate_data(tau=tau, time='1', normalize=normalize)
    model = LinearRegression()
    model.fit(X_train, y_train)
    if normalize:
        predictions = model.predict(X_valid)
        predictions = data.scaler.transform(predictions)
    else:
        predictions = model.predict(X_valid)
    print("Root Mean Squared Error: " + str(mean_squared_error(predictions, y_valid, squared=False)))


train_and_predict('../traffic.txt', tau=12)
