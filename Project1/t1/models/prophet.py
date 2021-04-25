import numpy as np
from joblib import Parallel, delayed
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error
from fbprophet import Prophet
from utils.preprocessing import *


def train_model(X: np.ndarray):
    model = Prophet()
    model.fit(X)
    return model


def train_and_predict(path: str):
    data = pd.read_csv(path, header=None, delimiter=',')
    t, num_models = data.shape
    train_size = 0.8
    train_num = int(t * train_size)
    test_num = t - train_num
    y_valid = data.values[train_num:].T
    base = datetime.strptime('2010-01-01', '%Y-%m-%d')
    date = [base + timedelta(hours=x) for x in range(train_num)]
    models = Parallel(n_jobs=4)(delayed(train_model)(pd.DataFrame({'ds': date, 'y': data.iloc[:train_num, i]})) for i in range(num_models))
    predictions = np.array([models[i].predict(models[i].make_future_dataframe(periods=test_num, freq='H'))['yhat'][int(train_size * t):] for i in range(num_models)])
    print("Root Mean Squared Error: " + str(mean_squared_error(predictions, y_valid, squared=False)))


train_and_predict('../traffic.txt')
