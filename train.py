import pandas

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import numpy as np


def train_model():
    """Not actually used for counting forecast, but used to train model and see how it differs from the actual"""
    datafile_name = 'data-files/City_MedianListingPrice_2Bedroom.csv'
    file_data = pandas.read_csv(datafile_name, encoding='latin-1')
    file_data = file_data.set_index('RegionName')
    series = file_data.loc['Philadelphia', file_data.columns[5:]].dropna()
    series.Timestamp = pandas.to_datetime(series.index, format='%Y-%m')
    series.index = series.Timestamp
    series.index.freq = 'MS'

    train, test = train_test_split(series, test_size=0.2, shuffle=False)

    model = ExponentialSmoothing(np.asarray(train), seasonal='mul', trend='add', seasonal_periods=12).fit()
    predict = model.forecast(len(test))
    plt.plot(train.index, train, label='Train')
    plt.plot(test.index, test, label='Test')
    plt.plot(test.index, predict, label='Predict')
    plt.legend()
    plt.show()
