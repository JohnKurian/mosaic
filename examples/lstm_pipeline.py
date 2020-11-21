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
    print('im hereeeeeeee')
    dataset = pd.read_csv('weather_energy_hourly.csv')
    print('im hereeeeeeee 222222222222')
    dataset = dataset['pressure', 'dewPoint', 'avg_energy']
    # dataset = impute_columns(dataset)
    print('here too')

    reframed = build_lagged_features(dataset, ['pressure', 'dewPoint'], 'avg_energy', [1,1, cfg['lag_days']])
    reframed = scale(reframed)
    train, test = split_into_train_test(reframed, 49)
    train_X, train_y, test_X, test_y = prepare_x_and_y(train, test)
    model = construct_model(train_X, train_y, cfg['hidden_neurons'], cfg['batch_size'])
    predictions = predict(model, test_X)
    rmse, r2 = calculate_loss(predictions, test_y)
    print(rmse, r2)
    return rmse








