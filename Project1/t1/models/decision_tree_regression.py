from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from utils.preprocessing import *


X, y = generate_data('../commodity.txt', tau=7)
X_train, X_valid, y_train, y_valid = train_test_split(X, y)
num_models = X_train.shape[0]
model = DecisionTreeRegressor(random_state=1)
model.fit(X_train, y_train)
predictions = model.predict(X_valid)
# models = list()
# predictions = list()
# for i in range(num_models):
#     model = DecisionTreeRegressor(random_state=1)
#     model.fit(X_train[i], y_train[i])
#     models.append(model)
#     predictions.append(model.predict(X_valid[i]))
print("Root Mean Squared Error: " + str(mean_squared_error(predictions, y_valid, squared=False)))
