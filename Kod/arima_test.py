from statsmodels.tsa.arima_model import ARIMA


# Function that calls ARIMA model to fit and forecast the data
def StartARIMAForecasting(Actual, P, D, Q):
	model = ARIMA(Actual, order=(P, D, Q))
	model_fit = model.fit(disp=0)
	prediction = model_fit.forecast()[0]
	return prediction


# creating data
ExchangeRates = [[1],[1.5],[2],[4],[6],[8],[16],[1]]

# predict next value
predicted = StartARIMAForecasting(ExchangeRates, 1,1,0)
# display the value
print('Predicted=%f' % (predicted))