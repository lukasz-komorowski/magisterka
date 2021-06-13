from reading_data import *
import matplotlib.pyplot as plt
import pandas as pd
from finta import TA

# # # example of training a final regression model
# # from sklearn.linear_model import LinearRegression
# # from sklearn.datasets import make_regression
# #
# # # generate regression dataset
# # X, y = make_regression(n_samples=10, n_features=2, noise=0.1)
# # print(type(X))
# # print(X)
# # print(type(y))
# # print(y)
# # # fit final model
# # model = LinearRegression()
# # model.fit(X, y)
# # # define one new data instance
# # Xnew = [[-1.07296862, -0.52817175]]
# # # make a prediction
# # ynew = model.predict(Xnew)
# # # show the inputs and predicted outputs
# # print("X=%s, Predicted=%s" % (Xnew[0], ynew[0]))
#
# print(__doc__)
#
# import numpy as np
# from sklearn.svm import SVR
# import matplotlib.pyplot as plt
# from forecast_rates import *
#
# # #############################################################################
# # Generate sample data
# X = np.sort(5 * np.random.rand(40, 1), axis=0)
# y = np.sin(X).ravel()
#
# # #############################################################################
# # Add noise to targets
# y[::5] += 3 * (0.5 - np.random.rand(8))
#
# # #############################################################################
# # Fit regression model
# svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
# svr_lin = SVR(kernel='linear', C=100, gamma='auto')
# svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1,
#                coef0=1)
#
# # #############################################################################
# # Look at the results
# lw = 2
#
# svrs = [svr_rbf, svr_lin, svr_poly]
# kernel_label = ['RBF', 'Linear', 'Polynomial']
# model_color = ['m', 'c', 'g']
#
# fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 10), sharey=True)
# for ix, svr in enumerate(svrs):
#     axes[ix].plot(X, svr.fit(X, y).predict(X), color=model_color[ix], lw=lw,
#                   label='{} model'.format(kernel_label[ix]))
#     axes[ix].scatter(X[svr.support_], y[svr.support_], facecolor="none",
#                      edgecolor=model_color[ix], s=50,
#                      label='{} support vectors'.format(kernel_label[ix]))
#     axes[ix].scatter(X[np.setdiff1d(np.arange(len(X)), svr.support_)],
#                      y[np.setdiff1d(np.arange(len(X)), svr.support_)],
#                      facecolor="none", edgecolor="k", s=50,
#                      label='other training data')
#     axes[ix].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
#                     ncol=1, fancybox=True, shadow=True)
#
# fig.text(0.5, 0.04, 'data', ha='center', va='center')
# fig.text(0.06, 0.5, 'target', ha='center', va='center', rotation='vertical')
# fig.suptitle("Support Vector Regression", fontsize=14)
# plt.show()


# from sklearn import svm, datasets
# from sklearn.model_selection import GridSearchCV
# iris = datasets.load_iris()
# parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
# svc = svm.SVC()
# clf = GridSearchCV(svc, parameters)
# print(clf.fit(iris.data, iris.target))
#
#
# print(sorted(clf.cv_results_.keys()))


# exchange_rates = get_values_from_csv("NBP_dane_100.csv")
#
# df = pd.DataFrame(exchange_rates, columns=['value'])
# print(df)
#
# plt.plot(df.value, label='CHF')
# plt.show()
#
# exp1 = df.value.ewm(span=12, adjust=False).mean()
# exp2 = df.value.ewm(span=26, adjust=False).mean()
# macd = exp1-exp2
# exp3 = macd.ewm(span=9, adjust=False).mean()
#
# plt.plot(macd, label='AMD MACD', color = '#EBD2BE')
# plt.plot(exp3, label='Signal Line', color='#E5A4CB')
# plt.legend(loc='upper left')
# plt.show()

# ToDo: check FinTA (Financial Technical Analysis)
#  pip install finta
#  https://pypi.org/project/finta/

ohlc = pd.read_csv("NBP_dane_ohlc_2020.csv", index_col="date", parse_dates=True)
SMA15_CHF = TA.SMA(ohlc, 10)
SMA30_CHF = TA.SMA(ohlc, 30)
SMA45_CHF = TA.SMA(ohlc, 45)
plt.plot(ohlc["close"], label='CHF', color='black')
# plt.plot(SMA15_CHF, label='SMA 10-dniowe')
plt.plot(SMA30_CHF, label='SMA 30-dniowe')
# plt.plot(SMA45_CHF, label='SMA 45-dniowe')
plt.legend(loc='upper left')
plt.show()
