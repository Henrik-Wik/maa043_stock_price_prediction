#%%
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from preprocessing import *

df = download_data()

X, y, X_y, feature_names = create_features(df)

X_train, X_test, y_train, y_test = time_split(X,y)
# %%

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=2)),
    ('SVR',SVR(kernel='linear'))
])

pipeline.fit(X_train, y_train)

Train_pred = pipeline.predict(X_test)
# %%


