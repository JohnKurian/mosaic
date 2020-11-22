from mosaic.mosaic import Search
from env import Environment
import configuration_space
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import svm, datasets
import numpy as np
import os
import sys
import inspect

from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression

from autokeras import StructuredDataRegressor


currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)


import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, Activation
import math

from sklearn.metrics import r2_score






def get_features(df, predictor):
    cols = list(df.columns)
    cols.remove(predictor)
    return cols


def build_lagged_features(df, selected_features, predictor_column, feature_wise_lag_days):
    f = selected_features
    temp_selected_features = selected_features + [predictor_column]
    df = df[temp_selected_features]
    for idx, col in enumerate(df):
        if feature_wise_lag_days[idx] > 0:
            for lag in range(1, feature_wise_lag_days[idx] + 1):
                df[col + '_lag_' + str(lag)] = df[col].shift(lag)

    df = df.dropna()

    new_cols = list(df.columns)
    new_cols.remove(predictor_column)
    new_cols.append(predictor_column)

    df = df[new_cols]

    return df


def scale(reframed):
    scaler = MinMaxScaler(feature_range=(0, 1))
    reframed = scaler.fit_transform(reframed)
    return reframed


def split_into_train_test(reframed, forecasting_window):
    # split into train and test sets
    train = reframed[:(len(reframed)-forecasting_window), :]
    test = reframed[(len(reframed)-forecasting_window):len(reframed), :]
    return train, test


def prepare_x_and_y(train, test):
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]

    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

    return train_X, train_y, test_X, test_y


def prepare_2d_x_and_y(train, test):
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]

    train_X = train_X.reshape((train_X.shape[0], train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[1]))

    return train_X, train_y, test_X, test_y



def construct_model(train_X, train_y, hidden_neurons, batch_size):
    model = None
    model = Sequential()
    model.add(LSTM(hidden_neurons, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    history = model.fit(train_X, train_y, epochs=200, batch_size=batch_size, verbose=0, shuffle=False)
    return model



def train_model(model, train_X, train_y, batch_size, epochs):
    history = model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=False)



def predict(model, test_X):
    yhat = model.predict(test_X)
    return yhat


def calculate_loss(yhat, test_y):
    rmse = mean_squared_error(yhat, test_y) ** 0.5
    r2 = r2_score(test_y, yhat, multioutput = "variance_weighted")
    return rmse, r2


def run_lstm_pipeline(cfg):
    dataset = pd.read_csv('./examples/weather_energy_hourly.csv')
    dataset = dataset[['pressure', 'dewPoint', 'avg_energy']]
    # dataset = impute_columns(dataset)

    reframed = build_lagged_features(dataset, ['pressure', 'dewPoint'], 'avg_energy', [1,1, cfg['lag_days']])
    reframed = scale(reframed)
    train, test = split_into_train_test(reframed, 49)
    train_X, train_y, test_X, test_y = prepare_x_and_y(train, test)
    model = construct_model(train_X, train_y, cfg['hidden_neurons'], cfg['batch_size'])
    predictions = predict(model, test_X)
    rmse, r2 = calculate_loss(predictions, test_y)
    print(rmse, r2)
    return r2


def run_linear_reg_pipeline(cfg):
    dataset = pd.read_csv('./examples/weather_energy_hourly.csv')
    dataset = dataset[['pressure', 'dewPoint', 'avg_energy']]
    # dataset = impute_columns(dataset)

    reframed = build_lagged_features(dataset, ['pressure', 'dewPoint'], 'avg_energy', [1,1, cfg['lag_days']])
    reframed = scale(reframed)
    train, test = split_into_train_test(reframed, 49)
    train_X, train_y, test_X, test_y = prepare_2d_x_and_y(train, test)
    print('problem')
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    reg = LinearRegression().fit(train_X, train_y)
    r2 = reg.score(test_X, test_y)
    print(r2)
    return r2


def run_ridge_reg_pipeline(cfg):
    dataset = pd.read_csv('./examples/weather_energy_hourly.csv')
    dataset = dataset[['pressure', 'dewPoint', 'avg_energy']]
    # dataset = impute_columns(dataset)
    print('here too')

    reframed = build_lagged_features(dataset, ['pressure', 'dewPoint'], 'avg_energy', [1,1, cfg['lag_days']])
    reframed = scale(reframed)
    train, test = split_into_train_test(reframed, 49)
    train_X, train_y, test_X, test_y = prepare_2d_x_and_y(train, test)
    clf = Ridge(alpha=cfg['alpha'])
    clf.fit(train_X, train_y)
    r2 = clf.score(test_X, test_y)
    print(r2)
    return r2



def run_autokeras_pipeline(cfg):
    dataset = pd.read_csv('./examples/weather_energy_hourly.csv')
    dataset = dataset[['pressure', 'dewPoint', 'avg_energy']]
    # dataset = impute_columns(dataset)

    reframed = build_lagged_features(dataset, ['pressure', 'dewPoint'], 'avg_energy', [1,1, cfg['lag_days']])
    reframed = scale(reframed)
    train, test = split_into_train_test(reframed, 49)
    train_X, train_y, test_X, test_y = prepare_2d_x_and_y(train, test)

    search = StructuredDataRegressor(max_trials=10, loss='mean_squared_error')
    search.fit(train_X, train_y, validation_data=(test_X, test_y))

    predictions = search.predict(test_X)
    rmse, r2 = calculate_loss(predictions, test_y)
    print(rmse, r2)
    return r2

def run_pipeline(cfg):

    r2 = 0

    if cfg['algo'] == 'lstm':
        r2 = run_lstm_pipeline(cfg)
    elif cfg['algo'] == 'autokeras':
        r2 = run_autokeras_pipeline(cfg)
    elif cfg['algo'] == 'linear_reg':
        r2 = run_linear_reg_pipeline(cfg)
    elif cfg['algo'] == 'ridge_reg':
        r2 = run_ridge_reg_pipeline(cfg)

    return r2











def svm_from_cfg(cfg):
    """ Creates a SVM based on a configuration and evaluates it on the
    iris-dataset using cross-validation.
    Source: https://automl.github.io/SMAC3/master/examples/SMAC4HPO_svm.html

    Parameters:
    -----------
    cfg: Configuration (ConfigSpace.ConfigurationSpace.Configuration)
        Configuration containing the parameters.
        Configurations are indexable!

    Returns:
    --------
    A crossvalidated mean score for the svm on the loaded data-set.
    """
    # For deactivated parameters, the configuration stores None-values.
    # This is not accepted by the SVM, so we remove them.
    cfg = {k: cfg[k] for k in cfg if cfg[k]}
    # We translate boolean values:
    cfg["shrinking"] = True if cfg["shrinking"] == "true" else False
    # And for gamma, we set it to a fixed value or to "auto" (if used)
    if "gamma" in cfg:
        cfg["gamma"] = cfg["gamma_value"] if cfg["gamma"] == "value" else "auto"
        cfg.pop("gamma_value", None)  # Remove "gamma_value"

    clf = svm.SVC(**cfg, random_state=42)

    scores = cross_val_score(clf, X_train, y_train, cv=5)
    return np.mean(scores)  # Minimize!



base_params = {
    'forecasting_window' : 49,
    'predictor_column' : 'avg_energy',

    'data_imputation' : 0,

    'feature_wise_lag_days' : [1,1,28],

    'normalize' : False,

    'hyperparameters': {
        'hidden_neurons' : [40, 70],
        'lag_days' : [3, 200],
        'batch_size' : [20, 120]
    },

    'hidden_neurons' : 50,
    'lag_days' : 10,
    'batch_size' : 50,

    'selected_features' : ['pressure', 'dewPoint'],

    'datafile_path' : 'weather_energy_hourly.csv',
    'model_type': 'lstm',
    'num_trials': 1000
}





if __name__ == "__main__":
    # iris = datasets.load_iris()
    # X_train, X_test, y_train, y_test = train_test_split(
    #     iris.data, iris.target, test_size=0.33, random_state=42)

    # environment = Environment(svm_from_cfg,
    #                           config_space=configuration_space.cs,
    #                           mem_in_mb=2048,
    #                           cpu_time_in_s=30,
    #                           seed=42)


    environment = Environment(run_pipeline,
                              config_space=configuration_space.cs,
                              mem_in_mb=8000,
                              cpu_time_in_s=300000000,
                              seed=42)

    mosaic = Search(environment=environment,
                    bandit_policy={"policy_name": "uct", "c_ucb": 1.1},
                    coef_progressive_widening=0.6,
                    verbose=True)
    best_config, best_score = mosaic.run(nb_simulation=10000)
    print("Best config: ", best_config, "best score", best_score)
