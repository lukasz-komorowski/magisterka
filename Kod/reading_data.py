from pandas import read_csv


def get_all_from_csv(fileName):
    # get columns with names from csv
    #         Data    Kurs
    # 0 2002-01-02  2.3915
    # 1 2002-01-03  2.4034
    # 2 2002-01-04  2.3895
    # 3 2002-01-07  2.3726
    # 4 2002-01-08  2.3813
    return read_csv(fileName, parse_dates=[0])


def get_data_from_csv(fileName):
    # get columns without names from csv
    # Data
    # 2002-01-02  2.3915
    # 2002-01-03  2.4034
    # 2002-01-04  2.3895
    # 2002-01-07  2.3726
    # 2002-01-08  2.3813
    return read_csv(fileName, parse_dates=[0], index_col=0)


def get_values_from_csv(fileName):
    # get only values from csv
    # [2.3915]
    # [2.4034]
    # [2.3895]
    # [2.3726]
    # [2.3813]
    return read_csv(fileName, header=0, parse_dates=[0], index_col=0).values
