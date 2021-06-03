import matplotlib as plot
import matplotlib.pyplot as plt
import numpy as np
from pandas import read_csv
from matplotlib import pyplot
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared
from sklearn.datasets import make_regression
from sklearn.model_selection import GridSearchCV


def get_all_from_csv(fileName):
    #get columns with names from csv
    #         Data    Kurs
    # 0 2002-01-02  2.3915
    # 1 2002-01-03  2.4034
    # 2 2002-01-04  2.3895
    # 3 2002-01-07  2.3726
    # 4 2002-01-08  2.3813
    return read_csv(fileName, parse_dates=[0])


def get_data_from_csv(fileName):
    #get columns without names from csv
    # Data
    # 2002-01-02  2.3915
    # 2002-01-03  2.4034
    # 2002-01-04  2.3895
    # 2002-01-07  2.3726
    # 2002-01-08  2.3813
    return read_csv(fileName, parse_dates=[0], index_col=0)


def get_values_from_csv(fileName):
    #get only values from csv
    # [2.3915]
    # [2.4034]
    # [2.3895]
    # [2.3726]
    # [2.3813]
    return read_csv(fileName, header=0, parse_dates=[0], index_col=0).values


def prepare_data_for_NN(raw_data, days):
    # print(raw_data)
    X_list = []
    Y_list = []
    for i in range(len(raw_data)-days):
        X_tmp = []
        for j in range(days):
            X_tmp.append(raw_data[i+j].tolist()[0])
        X_list.append(X_tmp)
        Y_list.append(raw_data[i+days].tolist()[0])
    # print(X_list)
    # print(Y_list)
    X_list = np.asarray(X_list)
    Y_list = np.asarray(Y_list)
    return (X_list, Y_list)


def prepare_prediction_data(X_list, Y_list):
    to_predict = X_list[len(X_list)-1]
    to_predict = np.delete(to_predict, 0)
    to_predict = np.append(to_predict, Y_list[len(Y_list)-1])
    to_predict = to_predict[np.newaxis, :]
    return to_predict


def print_rate_chart(exchangeRatesSeries):
    print(exchangeRatesSeries.head(10))
    print(exchangeRatesSeries.describe())
    plt.plot(exchangeRatesSeries)
    plt.show()
    plt.hist(exchangeRatesSeries)
    plt.show()


def start_arima_forecasting(Actual, P, D, Q):
    model = ARIMA(Actual, order=(P, D, Q))
    model_fit = model.fit()
    prediction = model_fit.forecast()
    print(prediction)
    prediction = model_fit.forecast()[0]
    return prediction


def prediction_arima_model(exchangeRatesValues):
    # Get exchange rates
    ActualData = exchangeRatesValues
    # Size of exchange rates
    NumberOfElements = len(ActualData)

    # Use 70% of data as training, rest 30% to Test model
    TrainingSize = int(NumberOfElements - 15)
    TrainingData = ActualData[0:TrainingSize]
    TestData = ActualData[TrainingSize:NumberOfElements]

    # new arrays to store actual and predictions
    Actual = [x for x in TrainingData]
    Predictions = list()

    # in a for loop, predict values using ARIMA model
    for timepoint in range(len(TestData)):
        ActualValue = TestData[timepoint]
        # forcast value
        Prediction = start_arima_forecasting(Actual, 1, 1, 0)
        print('Actual=%f, Predicted=%f' % (ActualValue, Prediction))
        # add it in the list
        # print(type(ActualValue))
        # print(type(Prediction))
        Predictions.append(Prediction)
        Actual.append([Prediction])

    # Print MSE to see how good the model is
    Error = mean_squared_error(TestData, Predictions)
    print('Test Mean Squared Error (smaller the better fit): %.6f' % Error)

    Predictions.pop(0)
    TestData = list(TestData)
    TestData.pop()

    # plot
    pyplot.plot(TestData)
    pyplot.plot(Predictions, color='red')
    print('Actual length = %f, Predicted length = %f' % (len(TestData), len(Predictions)))
    pyplot.show()


def prediction_statistic(exchangeRates):
    statistic_weekly(exchangeRates)
    statistic_monthly(exchangeRates)


def statistic_weekly(exchangeRates):
    #which dasy of the week is the cheapest
    min_day = exchangeRates.iloc[0][0].weekday()
    min_val = exchangeRates.iloc[0][1]
    min_frq = [0, 0, 0, 0, 0]
    for day in range(len(exchangeRates)):
        current_day = exchangeRates.iloc[day][0].weekday()
        current_rate = exchangeRates.iloc[day][1]
        if current_day == 0:
            min_frq[min_day] += 1
            min_val = 99
        if current_rate < min_val:
            min_val = current_rate
            min_day = current_day

    print(min_frq)


def statistic_monthly(exchangeRates):
    #which dasy of the month is the cheapest
    min_day = exchangeRates.iloc[0][0].date().day
    print(min_day)
    min_val = exchangeRates.iloc[0][1]
    min_frq = [0]*31
    for day in range(len(exchangeRates)):
        current_day = exchangeRates.iloc[day][0].date().day
        current_rate = exchangeRates.iloc[day][1]
        if current_day == 1:
            min_frq[min_day-1] += 1
            min_val = 99
        if current_rate < min_val:
            min_val = current_rate
            min_day = current_day

    print(min_frq)


def statistic_weeks(exchangeRates):
    #which week of the year is the cheapest
    print(0)



def main():
    # exchangeRatesSeries = get_data_from_csv('NBP_dane.csv')
    # print(exchangeRatesSeries.head())
    exchangeRatesValues = get_values_from_csv('NBP_dane.csv')
    # for x in range(10):
    #     print(exchangeRatesValues[x])
    # exchangeRates = get_all_from_csv('NBP_dane.csv')
    # print(exchangeRates.head())

    (X, Y) = prepare_data_for_NN(exchangeRatesValues[:-1], 50)
    predict = prepare_prediction_data(X,Y)
    # print(type(X))
    # print(X.shape)
    # print(X)
    # print(type(Y))
    # print(Y.shape)
    # print(Y)
    # print(type(predict))
    # print(predict)

    # ToDo:
    #  1:
    #   svm.NuSVR
    #   neural_network.MLPRegressor
    #   gaussian_process.GaussianProcessRegressor
    #   tree.DecisionTreeRegressor
    #   tree.ExtraTreeRegressor
    #  2
    #   linear_model.RidgeCV
    #   linear_model.SGDRegressor
    #   linear_model.BayesianRidge
    #  3:
    #   cross_decomposition.PLSRegression


    reg_SVR = GridSearchCV(SVR(kernel='rbf', gamma=0.1), \
                        param_grid={"C": [1e0, 1e1, 1e2, 1e3], \
                                    "gamma": np.logspace(-2, 2, 5)}).fit(X, Y)
    prediction_SVR = reg_SVR.predict(predict)
    score_SVR = reg_SVR.score(X, Y)
    print(reg_SVR.get_params())
    print("X = %s, Prediction = %s, score = %s" % (predict[0], prediction_SVR[0], score_SVR))


    reg_KR = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1), \
                        param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3], \
                                    "gamma": np.logspace(-2, 2, 5)}).fit(X, Y)
    prediction_KR = reg_KR.predict(predict)
    score_KR = reg_KR.score(X, Y)
    print(reg_KR.get_params())
    print("X = %s, Prediction = %s, score = %s" % (predict[0], prediction_KR[0], score_KR))

    gp_kernel = ExpSineSquared(1.0, 5.0, periodicity_bounds=(1e-2, 1e1)) + WhiteKernel(1e-1)
    reg_gpr = GaussianProcessRegressor(kernel=gp_kernel).fit(X, Y)
    prediction_gpr = reg_gpr.predict(predict)
    score_gpr = reg_gpr.score(X, Y)
    print(reg_gpr.get_params())
    print("X = %s, Prediction = %s, score = %s" % (predict[0], prediction_gpr[0], score_gpr))

    # model = LinearRegression()
    # model.fit(X, Y)
    # prediction = model.predict(predict)
    # print("X = %s, Prediction = %s" % (predict[0], prediction[0]))

    
    # print_rate_chart(exchangeRatesSeries)
    # prediction_arima_model(exchangeRatesValues)
    # prediction_statistic(exchangeRates)
    #statistic_monthly(exchangeRates)


if __name__ == "__main__":
    main()
