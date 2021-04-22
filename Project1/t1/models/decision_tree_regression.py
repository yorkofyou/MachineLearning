from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV
from utils.preprocessing import *


def train_and_predict(path: str):
    X, y = generate_data(path, tau=7)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y)
    num_models = X_train.shape[0]
    parameters = {'criterion': ['mse', 'mae'],
                  'max_depth': [2, 5, 10, 20],
                  'min_samples_split': [2, 3, 5],
                  'min_samples_leaf': [1, 5, 8],
                  'max_features': ['log2', 'sqrt', 'auto']}
    # model = DecisionTreeRegressor(random_state=1)
    # model.fit(X_train, y_train)
    # predictions = model.predict(X_valid)
    models = list()
    predictions = list()
    for i in range(num_models):
        model = DecisionTreeRegressor(random_state=1)
        grid_object = RandomizedSearchCV(model, parameters)
        grid_object.fit(X_train[i], y_train[i])
        models.append(grid_object)
        predictions.append(grid_object.predict(X_valid[i]))
    print("Root Mean Squared Error: " + str(mean_squared_error(predictions, y_valid, squared=False)))


train_and_predict('../commodity.txt')
