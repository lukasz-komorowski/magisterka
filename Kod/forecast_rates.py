# import pandas
import pandas as pa
import matplotlib as plot
import matplotlib.pyplot as plt

# get data
from pandas import read_csv


def GetData(fileName):
    return read_csv(fileName, parse_dates=[0], index_col=0)


# read time series from the exchange.csv file
exchangeRatesSeries = GetData('NBP_dane.csv')

# view top 10 records
print(exchangeRatesSeries.head(10))
print(exchangeRatesSeries.describe())

plt.plot(exchangeRatesSeries)
plt.show()
plt.hist(exchangeRatesSeries)
plt.show()
