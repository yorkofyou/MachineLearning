from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from utils.preprocessing import *


def train_and_predict(path: str, tau: int, normalize=False):
    data = Dataset(path)
    X_train, X_valid, y_train, y_valid = data.generate_data(tau=tau, time='1', normalize=normalize)
    parameters = {'n_estimators': [20, 50, 100, 200, 500],
                  'eval_metric': ['rmse', 'mae'],
                  'max_depth': [2, 5, 10],
                  'min_child_weight': [1, 3, 5]}
    model = XGBRegressor(random_state=1)
    grid_object = GridSearchCV(model, parameters)
    grid_object.fit(X_train, y_train)
    print(grid_object.best_params_)
    if normalize:
        predictions = model.predict(X_valid)
        predictions = data.scaler.transform(predictions)
    else:
        predictions = model.predict(X_valid)
    print("Root Mean Squared Error: " + str(mean_squared_error(predictions, y_valid, squared=False)))


train_and_predict('../electricity.txt', tau=24)
