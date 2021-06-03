import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.ar_model import AutoReg
from reading_data import *

# Function that calls ARIMA model to fit and forecast the data
def StartARIMAForecasting(Actual, P, D, Q):
	model = ARIMA(Actual, order=(P, D, Q))
	model_fit = model.fit()
	prediction = model_fit.forecast(steps=100000)
	return prediction


def StartARXForecasting(Actual, P):
	model = AutoReg(Actual, lags=P)
	model_fit = model.fit()
	prediction = model_fit.forecast(steps=100000)
	return prediction


# creating data
# ExchangeRates = [[1],[1.5],[2],[4],[6],[8],[16],[1]]
# ExchangeRates = [[1],[2],[4],[8],[16],[32],[64],[128]]
# ExchangeRates = [[1],[2],[4],[6],[8],[10],[14],[16]]
# ExchangeRates = get_values_from_csv('NBP_dane.csv')

rng = np.random.RandomState(0)
X = 5 * rng.rand(10000, 1)
X_plot = np.linspace(0, 5, 100000)[:, None]
ExchangeRates = np.sin(X).ravel()


# predict next value
predicted_ARIMA = StartARIMAForecasting(ExchangeRates, 2, 2, 1)
predicted_ARX = StartARXForecasting(ExchangeRates, 3)
# display the value
print('Predicted = %s \t %s' % (predicted_ARIMA, predicted_ARX))

y = ExchangeRates
plt.scatter(X, y, c='r', s=50, label='SVR support vectors',zorder=2, edgecolors=(0, 0, 0))
plt.scatter(X[:100], y[:100], c='k', label='data', zorder=1, edgecolors=(0, 0, 0))
plt.plot(X_plot, predicted_ARIMA, c='r', label='ARIMA')
plt.plot(X_plot, predicted_ARX, c='g', label='ARX')
plt.xlabel('data')
plt.ylabel('target')
plt.title('SVR versus Kernel Ridge')
plt.legend()