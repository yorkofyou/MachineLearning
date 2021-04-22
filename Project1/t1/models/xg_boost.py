from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from utils.preprocessing import *
from utils.plot import *


def train_and_predict(path: str):
    X, y = generate_data(path, tau=7)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y)
    num_models = X_train.shape[0]
    models = list()
    parameters = {'n_estimators': [100, 200, 300, 500, 1000]}
    predictions = list()
    for i in range(num_models):
        model = XGBRegressor(random_state=1)
        model.fit(X_train[i], y_train[i],
                  verbose=False)
        grid_object = GridSearchCV(model, parameters)
        grid_object.fit(X_train[i], y_train[i])
        models.append(grid_object)
        predictions.append(grid_object.predict(X_valid[i]))
    print("Root Mean Squared Error: " + str(mean_squared_error(predictions, y_valid, squared=False)))
    # plot_results(predictions, y_valid)


train_and_predict('../commodity.txt')
