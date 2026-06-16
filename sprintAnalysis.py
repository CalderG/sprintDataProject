import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly_express as px
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
import os
os.environ["KERAS_BACKEND"] = "torch"
from skforecast.plot import set_dark_theme
from skforecast.datasets import fetch_dataset
from skforecast.deep_learning import create_and_compile_model
from skforecast.deep_learning import ForecasterRnn
from skforecast.model_selection import TimeSeriesFold
from skforecast.model_selection import backtesting_forecaster_multiseries
from skforecast.plot import plot_prediction_intervals
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from feature_engine.datetime import DatetimeFeatures
from feature_engine.creation import CyclicalFeatures
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import kpss
#from sklearn.gaussian_process import GaussianProcessRegressor
#from sklearn.gaussian_process.kernels import RBF
import warnings
from statsmodels.tsa.api import ExponentialSmoothing, Holt, SimpleExpSmoothing


sprint_performances = pd.read_csv("C:/Users/calde/OneDrive/Documents/sprintData/analysis_ready_performances.csv",
                                   low_memory= False)

# Re-converting the date columns to datetime as the typing is lost when
# exported to csv


# Convert the date column and date of birth columns into date formats

sprint_performances["date"] = pd.to_datetime(sprint_performances["date"], format='mixed')

# Deal with duplicates by adding timedeltas

#sprint_performances['date'] = sprint_performances['date'] + pd.to_timedelta(sprint_performances.groupby('date').cumcount(), unit='ms')
 # + '20:00:00'

# Idea: For each athlete, sort by date and then add timedelta to all dates, thus resolving duplicates

# Then you can groupby quarterfinals, semi finals, heats, and finals


sprint_performances["dateOfBirth"] = pd.to_datetime(sprint_performances["dateOfBirth"], format = "mixed")

# Extract the year, month, and day of each performance

sprint_performances["yearOfResult"] = sprint_performances["date"].dt.year

sprint_performances["monthOfResult"] = sprint_performances["date"].dt.month

sprint_performances["dayOfResult"] = sprint_performances["date"].dt.day

# Make a new dataset looking at the fastest times for each week from 1948 to 2025
# Used Copilot to turn the dataset into just results and datetime indexes, and interpolate
# the results using time and then fill in any remaining NAs using backward and forward fill
weekly_minimum = sprint_performances.set_index("date").resample("W").min(numeric_only=True)

# Fill missing monthly values using time-aware interpolation and edge fill.
# This preserves the monthly frequency required for a traditional time series model.
weekly_minimum["time-corrected"] = weekly_minimum["time-corrected"].interpolate(method="time")
weekly_minimum["time-corrected"] = weekly_minimum["time-corrected"].bfill()
weekly_minimum["time-corrected"] = weekly_minimum["time-corrected"].ffill()

# Initial plot

betterVisual = px.line(weekly_minimum, x = weekly_minimum.index, y = "time-corrected")
betterVisual.show()


plt.plot(weekly_minimum["time-corrected"])
plt.title('Fastest Corrected Times every week from 1948')
plt.xlabel('Date')
plt.ylabel('Time in seconds')
plt.show()

# Test for stationarity 
# If the p value is less than 0.05, it rejects the null hypothesis
# that the time series is non stationary, thus it must be stationary
adf_test = adfuller(weekly_minimum["time-corrected"])
print('ADF Statistic: %f' % adf_test[0])
print('p-value: %f' % adf_test[1])


# P-value is 0.000015 < alpha = 0.05, so the time series is stationary, so no differencing term will be added
#monthly_minimum.sort_index(ascending=True, inplace=True)


# KPSS test for trend stationary, looking to see if
# the time series is stationary if the trend is removed
# Same decision rule as Augmented Dickey's Fuller Test
# The null hypothesis of KPSS is that the time series is trend stationary

kpss(weekly_minimum["time-corrected"], regression = 'ct')
# P-value is less than 0.01 which is less than 0.05, so the time series is not
# stationary when the trend is removed

# Trend will be dealt with by differencing in the ARIMA model

# ARIMA model first:

# Order p is the lag number where sharp dropoff occurs in PACF
# Q - number of forecast errors/noise terms to consider - look at ACF and find
# where it drops off sharply
# Lags are time delays between sprint performances
plot_acf(weekly_minimum['time-corrected'], lags=40, alpha = 0.05)
plot_pacf(weekly_minimum['time-corrected'], lags=40, alpha = 0.05)
plt.show()

# Divide data into training and testing
train_size = int(len(weekly_minimum) * 0.8)
weekly_train = weekly_minimum.iloc[:train_size]
weekly_test = weekly_minimum.iloc[train_size:]

# Traditional time series modeling on the evenly spaced weekly data
weekly_model = ARIMA(weekly_train["time-corrected"], order=(2, 1, 1))
weekly_model_fit = weekly_model.fit()

weekly_forecast = weekly_model_fit.get_forecast(steps=len(weekly_test))
weekly_forecast_series = pd.Series(weekly_forecast.predicted_mean, index=weekly_test.index)

mse = mean_squared_error(weekly_test["time-corrected"], weekly_forecast_series)
rmse = np.sqrt(mse)
print(f"Weekly ARIMA RMSE: {rmse:.4f}")

plt.figure(figsize=(14,7))
plt.plot(weekly_train["time-corrected"], label='Weekly Training Minimums')
plt.plot(weekly_test["time-corrected"], label='Weekly Test Minimums', color='orange')
plt.plot(weekly_forecast_series, label='Weekly ARIMA Forecast', color='green')
plt.fill_between(weekly_test.index,
                 weekly_forecast.conf_int().iloc[:, 0],
                 weekly_forecast.conf_int().iloc[:, 1],
                 color='k', alpha=.15)
plt.title('ARIMA Forecast on Weekly Fastest Sprint Times')
plt.xlabel('Date')
plt.ylabel('Time in seconds')
plt.legend()
plt.show()

# RMSE was 0.284 seconds for result column
# RMSE was 0.3116 seconds for time-corrected column

# Monthly test

monthly_minimum = sprint_performances.set_index("date").resample("ME").min(numeric_only=True)

# Fill missing monthly values using time-aware interpolation and edge fill.
# This preserves the monthly frequency required for a traditional time series model.
monthly_minimum["time-corrected"] = monthly_minimum["time-corrected"].interpolate(method="time")
monthly_minimum["time-corrected"] = monthly_minimum["time-corrected"].bfill()
monthly_minimum["time-corrected"] = monthly_minimum["time-corrected"].ffill()

# Initial plot

betterVisual = px.line(monthly_minimum, x = monthly_minimum.index, y = "time-corrected")
betterVisual.show()


plt.plot(monthly_minimum["time-corrected"])
plt.title('Fastest Corrected Times every month from 1948')
plt.xlabel('Date')
plt.ylabel('Time in seconds')
plt.show()

# Test for stationarity 
adf_test = adfuller(monthly_minimum["time-corrected"])
print('ADF Statistic: %f' % adf_test[0])
print('p-value: %f' % adf_test[1])

# P-value is 0.173513 > alpha = 0.05, so the time series is stationary, so no differencing term will be added

# For time-corrected, p-value is 0.331698 so the time series is not stationary

# KPSS test for trend stationary, looking to see if
# the time series is stationary if the trend is removed
# Null hypothesis is that the time series is trend statinoary

kpss(monthly_minimum["time-corrected"], regression = 'ct')

# Alpha was less than 0.01, so it rejected the null, therefore
# a differencing term will be added

# ARIMA model first:

# Order p is the lag number where sharp dropoff occurs in PACF
# Q - number of forecast errors/noise terms to consider - look at ACF and find
# where it drops off sharply
# Lags are time delays between sprint performances
plot_acf(monthly_minimum['time-corrected'], lags=40, alpha = 0.05)
plot_pacf(monthly_minimum['time-corrected'], lags=40, alpha = 0.05)
plt.show()

# Autocorrelation is showing seasonality, so while the ARIMA results are interesting,
# a SARIMA model will be tested as well
# Additionally, it cycles every 12 lags, so the seasonality order will have
# s = 12

# Divide data into training and testing
train_size = int(len(monthly_minimum) * 0.8)
monthly_train = monthly_minimum.iloc[:train_size]
monthly_test = monthly_minimum.iloc[train_size:]

# Traditional time series modeling on the evenly spaced monthly data
monthly_model = ARIMA(monthly_train["time-corrected"], order=(2, 1, 1))
monthly_model_fit = monthly_model.fit()

monthly_forecast = monthly_model_fit.get_forecast(steps=len(monthly_test))
monthly_forecast_series = pd.Series(monthly_forecast.predicted_mean, index=monthly_test.index)

mse2 = mean_squared_error(monthly_test["time-corrected"], monthly_forecast_series)
rmse2 = np.sqrt(mse2)
print(f"Monthly ARIMA RMSE: {rmse2:.4f}")

plt.figure(figsize=(14,7))
plt.plot(monthly_train["time-corrected"], label='Monthly Training Minimums')
plt.plot(monthly_test["time-corrected"], label='Monthly Test Minimums', color='orange')
plt.plot(monthly_forecast_series, label='Monthly ARIMA Forecast', color='green')
plt.fill_between(monthly_test.index,
                 monthly_forecast.conf_int().iloc[:, 0],
                 monthly_forecast.conf_int().iloc[:, 1],
                 color='k', alpha=.15)
plt.title('ARIMA Forecast on Monthly Fastest Sprint Times')
plt.xlabel('Date')
plt.ylabel('Time in seconds')
plt.legend()
plt.show()

# RMSE was 0.1966 seconds for result column
# RMSE was 0.2010 seconds for time-corrected column


# Monthly data had a lower RMSE, so the remaining time series models will stick with it

# SARIMA Test - Univariate 

sarima_univariate_model = SARIMAX(endog = monthly_train["time-corrected"], order=(2, 1, 1), seasonal_order = (2,0,1,12))
sarima_univariate_fit = sarima_univariate_model.fit()

sarima_univariate_forecast = sarima_univariate_fit.get_forecast(steps=len(monthly_test))
sarima_univariate_series = pd.Series(sarima_univariate_forecast.predicted_mean, index=monthly_test.index)

mse3 = mean_squared_error(monthly_test["time-corrected"], sarima_univariate_series)
rmse3 = np.sqrt(mse3)
print(f"Monthly SARIMA RMSE: {rmse3:.4f}")

plt.figure(figsize=(14,7))
plt.plot(monthly_train["time-corrected"], label='Monthly Training Minimums')
plt.plot(monthly_test["time-corrected"], label='Monthly Test Minimums', color='orange')
plt.plot(sarima_univariate_series, label='Monthly SARIMA Univariate Forecast', color='green')
plt.fill_between(monthly_test.index,
                 sarima_univariate_forecast.conf_int().iloc[:, 0],
                 sarima_univariate_forecast.conf_int().iloc[:, 1],
                 color='k', alpha=.15)
plt.title('SARIMA Univariate Forecast on Monthly Fastest Sprint Times')
plt.xlabel('Date')
plt.ylabel('Time in seconds')
plt.legend()
plt.show()

# RMSE with time-corrected and D & d= 1: 0.1219
# RMSE with time-corrected and D & d= 0: 0.1213
# RMSE with time-corrected and D = 0, d= 1: 0.112
# RMSE with time-corrected and D = 1, d= 0: 0.1125


# SARIMA Monthly Model with Exogenous/Extra Variables

# Fill missing monthly wind and altitude values using time-aware interpolation and edge fill.
# This preserves the monthly frequency required for a traditional time series model.
monthly_minimum["wind"] = monthly_minimum["wind"].interpolate(method="barycentric")
monthly_minimum["altitude"] = monthly_minimum["altitude"].interpolate(method="barycentric")
monthly_minimum["wind"] = monthly_minimum["wind"].bfill()
monthly_minimum["wind"] = monthly_minimum["wind"].ffill()
monthly_minimum["altitude"] = monthly_minimum["altitude"].bfill()
monthly_minimum["altitude"] = monthly_minimum["altitude"].ffill()

# Divide into training and testing
train_size = int(len(monthly_minimum) * 0.8)
monthly_train = monthly_minimum.iloc[:train_size]
monthly_test = monthly_minimum.iloc[train_size:]

sarima_exo_model = SARIMAX(endog = monthly_train["time-corrected"], exog = monthly_train[["wind", "altitude"]], order=(2, 0, 1), seasonal_order = (2,1,1,12))
sarima_exo_fit = sarima_exo_model.fit()

sarima_exo_forecast = sarima_exo_fit.get_forecast(steps=len(monthly_test), exog = monthly_test[["wind", "altitude"]])
sarima_exo_series = pd.Series(sarima_exo_forecast.predicted_mean, index=monthly_test.index)

mse5 = mean_squared_error(monthly_test["time-corrected"], sarima_exo_series)
rmse5 = np.sqrt(mse5)
print(f"Monthly SARIMA RMSE with Wind and Altitude as covariates: {rmse5:.4f}")

plt.figure(figsize=(14,7))
plt.plot(monthly_train["time-corrected"], label='Monthly Training Minimums with Wind and Altitude as covariates')
plt.plot(monthly_test["time-corrected"], label='Monthly Test Minimums with Wind and Altitude as covariates', color='orange')
plt.plot(sarima_exo_series, label='Monthly SARIMA Exo Forecast', color='green')
plt.fill_between(monthly_test.index,
                 sarima_exo_forecast.conf_int().iloc[:, 0],
                 sarima_exo_forecast.conf_int().iloc[:, 1],
                 color='k', alpha=.15)
plt.title('SARIMA Exo Forecast on Monthly Fastest Sprint Times')
plt.xlabel('Date')
plt.ylabel('Time in seconds')
plt.legend()
plt.show()

# RMSE: 1.086 with time-corrected, D & d = 0
# RMSE with time-corrected, D = 0, d = 1: 0.2504
# RMSE with time-corrected, D = 1, d = 0: 0.1327
# RMSE with time-corrected, D = 1, d = 1: 0.1859

# Examine summaries of the best performing models so far in order to create equations

# SARIMA Model with Wind and Altitude as exogenous variables, regular difference of 1 and no seasonal difference
print(sarima_exo_fit.summary())

# SARIMA Univariate Model with regular difference of 1, no seasonal difference
print(sarima_univariate_fit.summary())

# Slice off the first 11 years
slice_date = pd.to_datetime("1960-01-01")
monthly_minimum_slice = monthly_minimum[monthly_minimum.index > slice_date]

train_size = int(len(monthly_minimum_slice) * 0.8)
monthly_train = monthly_minimum_slice.iloc[:train_size]
monthly_test = monthly_minimum_slice.iloc[train_size:]

sarima_exo_model = SARIMAX(endog = monthly_train["time-corrected"], exog = monthly_train[["wind", "altitude"]], order=(2, 1, 1), seasonal_order = (2,1,1,12))
sarima_exo_fit = sarima_exo_model.fit()

sarima_exo_forecast = sarima_exo_fit.get_forecast(steps=len(monthly_test), exog = monthly_test[["wind", "altitude"]])
sarima_exo_series = pd.Series(sarima_exo_forecast.predicted_mean, index=monthly_test.index)

mse6 = mean_squared_error(monthly_test["time-corrected"], sarima_exo_series)
rmse6 = np.sqrt(mse6)
print(f"Monthly SARIMA RMSE with Wind and Altitude as covariates: {rmse5:.4f}")

plt.figure(figsize=(14,7))
plt.plot(monthly_train["time-corrected"], label='Monthly Training Minimums with Wind and Altitude as covariates')
plt.plot(monthly_test["time-corrected"], label='Monthly Test Minimums with Wind and Altitude as covariates', color='orange')
plt.plot(sarima_exo_series, label='Monthly SARIMA Exo Forecast', color='green')
plt.fill_between(monthly_test.index,
                 sarima_exo_forecast.conf_int().iloc[:, 0],
                 sarima_exo_forecast.conf_int().iloc[:, 1],
                 color='k', alpha=.15)
plt.title('SARIMA Exo Forecast on Monthly Fastest Sprint Times')
plt.xlabel('Date')
plt.ylabel('Time in seconds')
plt.legend()
plt.show()


# RMSE: with time-corrected, D & d = 0: 0.1325
# RMSE with time-corrected, D = 0, d = 1: 0.1325
# RMSE with time-corrected, D = 1, d = 0: 0.1325
# RMSE with time-corrected, D = 1, d = 1: 0.1325





# RMSE with slice from 1960 onward, same settings as lowest RMSE for previous exog: 0.1327

# Naive Forecasts, Exponential Smoothing, RNN, etc.

# RNN - Univariate, multistep forecasting
## First split into training, validation, and testing

weekly_train_size = int(len(weekly_minimum) * 0.8)
weekly_validation_split = int(len(weekly_minimum) * 0.9)
weekly_training = weekly_minimum[:weekly_train_size]
weekly_validation = weekly_minimum[weekly_train_size:weekly_validation_split]
weekly_testing = weekly_minimum[weekly_validation_split:]

model_rnn_multistep = create_and_compile_model(
            series          = weekly_minimum[["time-corrected"]],    # Training data
            levels          = ["time-corrected"],  # Target column (sprint times)
            lags            = 40,      # Number of lags to use as predictors
            steps           = 24,      # Number of steps to predict
            recurrent_layer = "LSTM",  # Type of recurrent layer ('LSTM', 'GRU', or 'RNN')
            recurrent_layers_kwargs = [{"activation": "tanh"}, {"activation": "elu"}], # Good activation function as it stretches across the y-axis, which is useful when the y-axis consists of times
            recurrent_units = [128, 64],    # Number of units in the recurrent layer
            compile_kwargs  = {"optimizer": "adam", "loss" : "mse"}, # Configuring optimizer and loss function
            dense_units     = [64,32]       # Number of units in the dense layer
        )

model_rnn_multistep.summary()

# Constructing the Forecaster

weekly_forecaster = ForecasterRnn(
    estimator=model_rnn_multistep,
    levels=["time-corrected"],
    lags=40, # Number of lags
    transformer_series=MinMaxScaler(), # Scales down outliers
    fit_kwargs={
        "epochs": 25, # Number of training reps where forward and backward propagation occurs
        "batch_size": 512,  # Number of training samples
        "callbacks": [
            EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True) # Stops early after 3 epochs if validation does not improve, returns best weights at the time of lowest validation
        ],  # Callback to stop training when it is no longer learning.
        "series_val": weekly_validation,  # Validation data for model training.
    },
)

# Fit forecaster
weekly_forecaster.fit(weekly_training[["time-corrected"]])

# Graph of Validation and Training Loss
fig, ax = plt.subplots(figsize=(8, 3))
loss_plots = weekly_forecaster.plot_history(ax=ax)

# Predict using the forecaster
timeCorrected_predictions = weekly_forecaster.predict()

# Update the forecaster's epochs to take the epoch right before the best metrics
weekly_forecaster.set_fit_kwargs(
    {"epochs": 25, "batch_size": 512, "verbose": 1} # Same batch size and information about model, reduced epochs
)

# Backtesting  - uses error to improve model

cv = TimeSeriesFold(
         steps = weekly_forecaster.max_step,
         initial_train_size = int(len(weekly_minimum) * 0.8),  # Training + Validation Data
         refit = True # Similar to cross validation, but only on the time side
     )

metrics, predictions = backtesting_forecaster_multiseries(
    forecaster = weekly_forecaster,
    series = weekly_minimum[["time-corrected"]],
    cv = cv,
    levels = weekly_forecaster.levels, # Column to be predicted, this just refers to the column from the weekly forecaster
    metric = "mean_squared_error", # Error metric, not the same as ARIMA and SARIMAX for consistency but can be transformed to be the same
    verbose = True,
    suppress_warnings = True
)

metric_single_series = metrics.loc[metrics["levels"] == "time-corrected", "mean_squared_error"].iat[0] # Faster way to get a scalar MSE
metrics

recurrent_nn_rmse = np.sqrt(metric_single_series)
print(recurrent_nn_rmse) 



fig, ax = plt.subplots(figsize=(8, 3))
weekly_testing["time-corrected"].plot(ax=ax, label="actual_times")
predictions.loc[predictions["level"] == "time-corrected", "pred"].plot(ax=ax, label="predicted_times")
ax.set_title("Recurrent Neural Network Forecast vs Reality")
ax.legend();

# Things to change/do differently: change validation size to 10%, disable refitting, configure number of units,
# add more activation functions, add more variables (multivariate), add more lags?
# Maybe change to 5th epoch instead of 4th

# Before any tests, RMSE was 0.222, loss of 0.0096
# Test 1: Change validation size to 10%, leaving 20% for testing
# Initial validation loss before backtesting, 0.0364 at Epoch 6, will use epoch 5. RMSE Result: 0.210
# Test 2: Disable refitting, revert change from test 1. Result: RMSE of 0.295
# Test 3: Configured number of recurrent units to 128, reverted previous tests, RMSE:0.213
# Test 4: Revert previous tests, configure lags to 40. Result: RMSE of 0.207
# Test 5: Revert previous tests, add an exponential linear unit function to activation functions and a second layer of 64 units
# to correspond to it. RMSE: 0.174, had epoch = 24 due to exceptional lowest loss late in training
# Test 6: Combo of Test 4 and 5 (40 lags + elu function), RMSE: 0.169
# Test 7: Final Test (40 lags + elu function + 80-10-10 training, validation, testing)
# Initial validation loss of 0.0247, start at epoch with best val loss
# RMSE: 0.163
# Test 8: Test 7 but added 32 units to dense units [64,32], validation loss of 0.0235
# RMSE: 0.177

# RNN Forecast for Monthly Data

monthly_train_size = int(len(monthly_minimum) * 0.7)
monthly_validation_split = int(len(monthly_minimum) * 0.8)
monthly_training = monthly_minimum[:monthly_train_size]
monthly_validation = monthly_minimum[monthly_train_size:monthly_validation_split]
monthly_testing = monthly_minimum[monthly_validation_split:]

model_rnn_multistep = create_and_compile_model(
            series          = monthly_minimum[["time-corrected"]],    # Training data
            levels          = ["time-corrected"],  # Target column (sprint times)
            lags            = 40,      # Number of lags to use as predictors
            steps           = 24,      # Number of steps to predict
            recurrent_layer = "GRU",  # Type of recurrent layer ('LSTM', 'GRU', or 'RNN')
            recurrent_layers_kwargs = [{"activation": "tanh"}, {"activation": "elu"}], # Good activation function as it stretches across the y-axis, which is useful when the y-axis consists of times
            recurrent_units = [128, 64],    # Number of units in the recurrent layer
            compile_kwargs  = {"optimizer": "adam", "loss" : "mse"}, # Configuring optimizer and loss function
            dense_units     = [64,32]       # Number of units in the dense layer
        )

model_rnn_multistep.summary()

# Constructing the Forecaster

monthly_forecaster = ForecasterRnn(
    estimator=model_rnn_multistep,
    levels=["time-corrected"],
    lags=40, # Number of lags
    transformer_series=MinMaxScaler(), # Scales down outliers
    fit_kwargs={
        "epochs": 25, # Number of training reps where forward and backward propagation occurs
        "batch_size": 512,  # Number of training samples
        "callbacks": [
            EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True) # Stops early after 3 epochs if validation does not improve, returns best weights at the time of lowest validation
        ],  # Callback to stop training when it is no longer learning.
        "series_val": monthly_validation,  # Validation data for model training.
    },
)

# Fit forecaster
monthly_forecaster.fit(monthly_train[["time-corrected"]])

# Graph of Validation and Training Loss
fig, ax = plt.subplots(figsize=(8, 3))
loss_plots = monthly_forecaster.plot_history(ax=ax)

# Predict using the forecaster
timeCorrected_predictions = monthly_forecaster.predict()

# Update the forecaster's epochs to take the epoch right before the best metrics
monthly_forecaster.set_fit_kwargs(
    {"epochs": 20, "batch_size": 512, "verbose": 0} # Same batch size and information about model, reduced epochs
)

# Backtesting  - uses error to improve model

cv = TimeSeriesFold(
         steps = monthly_forecaster.max_step,
         initial_train_size = int(len(monthly_minimum) * 0.8),  # Training + Validation Data
         refit = True # Similar to cross validation, but only on the time side
     )

metrics, predictions = backtesting_forecaster_multiseries(
    forecaster = monthly_forecaster,
    series = monthly_minimum[["time-corrected"]],
    cv = cv,
    levels = monthly_forecaster.levels, # Column to be predicted, this just refers to the column from the weekly forecaster
    metric = "mean_squared_error", # Error metric, not the same as ARIMA and SARIMAX for consistency but can be transformed to be the same
    verbose = False,
    suppress_warnings = True
)

metric_single_series = metrics.loc[metrics["levels"] == "time-corrected", "mean_squared_error"].iat[0] # Faster way to get a scalar MSE
metrics

recurrent_nn_rmse = np.sqrt(metric_single_series)
print(recurrent_nn_rmse) 

# Forecasting the RNN for Monthly Fastest Times - 20% validation

fig, ax = plt.subplots(figsize=(8, 3))
monthly_testing["time-corrected"].plot(ax=ax, label="actual_times")
predictions.loc[predictions["level"] == "time-corrected", "pred"].plot(ax=ax, label="predicted_times")
ax.set_title("Recurrent Neural Network Forecast vs Reality")
ax.legend();

# RMSE: 0.199

# Exponential Smoothing Model:
# Getting data setup
monthly_train_size = int(len(monthly_minimum) * 0.8)
monthly_train = monthly_minimum[:monthly_train_size]
monthly_test = monthly_minimum[monthly_train_size:]
exponential_model = ExponentialSmoothing(endog = monthly_train["time-corrected"], seasonal_periods = 12)
exponential_fit = exponential_model.fit()
exponential_forecast = exponential_fit.forecast(steps=len(monthly_test))
exponential_series = pd.Series(exponential_forecast.values, index=monthly_test.index)
mse7 = mean_squared_error(monthly_test["time-corrected"], exponential_series)
rmse7 = np.sqrt(mse7)
print(f"Monthly Exponential Smoothing RMSE: {rmse5:.4f}")

plt.figure(figsize=(14,7))
plt.plot(monthly_train["time-corrected"], label='Monthly Training Minimums')
plt.plot(monthly_test["time-corrected"], label='Monthly Test Minimums', color='orange')
plt.plot(exponential_series, label='Monthly Exponential Smoothing Forecast', color='green')
plt.fill_between(monthly_test.index,
                 exponential_forecast.conf_int().iloc[:, 0],
                 exponential_forecast.conf_int().iloc[:, 1],
                 color='k', alpha=.15)
plt.title('Exponential Smoothing Forecast on Monthly Fastest Sprint Times')
plt.xlabel('Date')
plt.ylabel('Time in seconds')
plt.legend()
plt.show()

# RMSE: 0.1324