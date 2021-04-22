import pickle
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from utils.preprocessing import *


def train_and_predict(path: str):
    X, y = generate_data(path, tau=7)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y)
    model = RandomForestRegressor(random_state=1, n_jobs=4)
    parameters = {'n_estimators': [10, 50, 100, 150, 200],
                  'criterion': ['mse', 'mae'],
                  'max_depth': [2, 5, 10, 20],
                  'min_samples_split': [2, 3, 5],
                  'min_samples_leaf': [1, 5, 8],
                  'max_features': ['log2', 'sqrt', 'auto']}
    grid_object = RandomizedSearchCV(model, parameters)
    grid_object.fit(X_train, y_train)
    predictions = grid_object.predict(X_valid)
    filename = 'random_forest.sav'
    pickle.dump(grid_object, open(filename, 'wb'))
    print("Root Mean Squared Error: " + str(mean_squared_error(predictions, y_valid, squared=False)))
    scores = cross_val_score(grid_object, np.concatenate((X_train, X_valid)), np.concatenate((y_train, y_valid)),
                             cv=5,
                             scoring='neg_root_mean_squared_error')
    print("Root Mean Squared Error using cross validation: " + scores)
