# %%
from preprocessing import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

df = download_data()

X, y, X_y, feature_names = create_features(df)

X_train, X_test, y_train, y_test = time_split(X, y)

scaled_X_train, scaled_X_test, scaler = scale_data(
    X_train, X_test, y_train.values.reshape(-1, 1))
# %%

regressors = [
    RandomForestRegressor(),
    LinearRegression(),
    SVR(),
    KNeighborsRegressor()
]

for regressor in regressors:
    reg = regressor.fit(scaled_X_train, y_train)
    train_pred = reg.predict(X_train)
    test_pred = reg.predict(X_test)
    train_acc = mean_absolute_error(y_train, train_pred)
    test_acc = mean_absolute_error(y_test, test_pred)
    print(f"{regressor.__class__.__name__}: {train_acc}")
    print(f"{regressor.__class__.__name__}: {test_acc}")
# %%
