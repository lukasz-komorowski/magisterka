import matplotlib as plot
import matplotlib.pyplot as plt
import numpy as np
from pandas import read_csv
from matplotlib import pyplot
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.datasets import make_regression
from reading_data import *


def prediction_statistic(exchangeRates):
    statistic_weekly(exchangeRates)
    statistic_monthly(exchangeRates)
    statistic_weeks(exchangeRates)


def statistic_weekly(exchangeRates):
    # which day of the week is the cheapest
    min_frq = np.zeros((5, 5)) # place, day
    current_week = np.zeros((5, 2))
    for day in range(len(exchangeRates)):
        current_day = exchangeRates.iloc[day][0].weekday()
        current_rate = exchangeRates.iloc[day][1]
        print(current_day, current_rate)

        if current_day == 0: # new week
            sorted_table = False
            while not sorted_table:
                sorted_table = True
                for i in range(4):
                    if current_week[i][0] > current_week[i+1][0]:
                        sorted_table = False
                        tmp_rate = current_week[i+1][0]
                        tmp_day = current_week[i+1][1]
                        current_week[i + 1][0] = current_week[i][0]
                        current_week[i + 1][1] = current_week[i][1]
                        current_week[i][0] = tmp_rate
                        current_week[i][1] = tmp_day

            for i in range(5):
                if current_week[i][0] != 0:
                    min_frq[i][np.int(current_week[i][1])] += 1
            current_week = np.zeros((5, 2))

        current_week[current_day][0] = current_rate
        current_week[current_day][1] = current_day

    print(min_frq)
    # ToDo: brakuje jednego tygodnia - poprawić


def statistic_monthly(exchangeRates):
    # which day of the month is the cheapest
    min_day = exchangeRates.iloc[0][0].date().day
    min_val = exchangeRates.iloc[0][1]
    min_frq = [0] * 31
    for day in range(len(exchangeRates)):
        current_day = exchangeRates.iloc[day][0].date().day
        current_rate = exchangeRates.iloc[day][1]
        if current_day == 1:
            min_frq[min_day - 1] += 1
            min_val = 99
        if current_rate < min_val:
            min_val = current_rate
            min_day = current_day

    print(min_frq)


def statistic_weeks(exchangeRates):
    # which week of the year is the cheapest
    min_frq = [0] * 53
    current_week = exchangeRates.iloc[0][0].date().isocalendar()[1]
    week_length = 0
    accumulated_rate = 0
    min_mean_rate = 99
    min_week_number = 0
    for day in range(len(exchangeRates)):
        if current_week != exchangeRates.iloc[day][0].date().isocalendar()[1]: # new week begining
            mean_rate = accumulated_rate/week_length
            accumulated_rate = 0
            week_length = 0

            if mean_rate < min_mean_rate:  # new min mean value
                min_mean_rate = mean_rate
                min_week_number = current_week

            if exchangeRates.iloc[day][0].date().isocalendar()[1] == 1:  # new year
                min_frq[min_week_number - 1] += 1
                print(str(exchangeRates.iloc[day][0].date().isocalendar()[0] - 1) + ": " \
                      + str(min_mean_rate) + \
                      ", KW: " + str(min_week_number))
                min_mean_rate = 99

            current_week = exchangeRates.iloc[day][0].date().isocalendar()[1]

        accumulated_rate += exchangeRates.iloc[day][1]
        week_length += 1

    # last year
    mean_rate = accumulated_rate / week_length
    if mean_rate < min_mean_rate:  # new min mean value
        min_mean_rate = mean_rate
        min_week_number = current_week

    print(min_frq)



def main():
    # exchangeRates = get_all_from_csv('NBP_dane.csv')
    exchangeRates = get_all_from_csv('NBP_dane.csv')
    prediction_statistic(exchangeRates)


if __name__ == "__main__":
    main()
