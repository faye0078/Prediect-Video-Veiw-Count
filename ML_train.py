from sklearn.svm import SVR, LinearSVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
import numpy as np

# train data
def read_data():
    user = np.load('../data/mod_social.npy')
    aural = np.load('../data/mod_aural.npy')
    visual = np.load('../data/mod_visual.npy')
    text = np.load('../data/mod_textual.npy')
    X = np.concatenate([user, aural, visual, text], axis=1)
    target = np.load('../data/train.npy')

    # test data
    user_test = np.load('../data/mod_social_test.npy')
    aural_test = np.load('../data/mod_aural_test.npy')
    visual_test = np.load('../data/mod_visual_test.npy')
    text_test = np.load('../data/mod_textual_test.npy')
    X_test = np.concatenate([user_test, aural_test, visual_test, text_test], axis=1)

    return X, target, X_test

def main(model_name):
    X, target, X_test = read_data()
    if model_name == 'linear':
        linear = LinearRegression(fit_intercept=True, normalize=True)
        linear.fit(X, target)
        out_put = linear.predict(X_test)
        np.save('./run/ML/linear_output.npy', out_put)

    if model_name == 'knn':
        knn = KNeighborsRegressor(weights='distance', n_neighbors=8)
        knn.fit(X, target)
        out_put = knn.predict(X_test)
        np.save('./run/ML/knn_output.npy', out_put)

    if model_name == 'dtr':
        dtr = DecisionTreeRegressor()
        dtr.fit(X, target)
        out_put = dtr.predict(X_test)
        np.save('./run/ML/dtr_output.npy', out_put)

    if model_name == 'svr':
        svr = SVR(kernel='rbf')
        svr = MultiOutputRegressor(svr)
        svr.fit(X, target)
        out_put = svr.predict(X_test)
        np.save('./run/ML/svr_output.npy', out_put)

    if model_name == 'linear_svr':
        svr = LinearSVR()
        svr = MultiOutputRegressor(svr)
        svr.fit(X, target)
        out_put = svr.predict(X_test)
        np.save('./run/ML/linear_svr_output.npy', out_put)

    if model_name == 'rfr':
        rfr = RandomForestRegressor()
        rfr = rfr.fit(X, target)
        out_put = rfr.predict(X_test)
        np.save('./run/ML/rfr_output.npy', out_put)

    if model_name == 'adaboost':
        rng = np.random.RandomState(1)
        regr = AdaBoostRegressor(DecisionTreeRegressor(max_depth=10),
                                 n_estimators=100, random_state=rng)
        regr = MultiOutputRegressor(regr)
        regr = regr.fit(X, target)
        out_put = regr.predict(X_test)
        np.save('./run/ML/adaboost_output.npy', out_put)

if __name__ == "__main__":
    # linear knn dtr svr linear_svr rfr adaboost
    model_name = 'linear_svr'
    main(model_name)