import pandas
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import numpy as np


class HomePriceRentCalculator(object):
    CITY = 'city'
    ZIP = 'zip'

    BEDROOM_AMOUNT_FILE_NAMES = {
        '1': '1Bedroom.csv',
        '2': '2Bedroom.csv',
        '3': '3Bedroom.csv',
        '4': '4Bedroom.csv',
        '5+': '5BedroomOrMore.csv'
    }

    def __init__(self):
        self.bedrooms_amount = input('Enter bedrooms amount (1-4, 5+):')
        self.location_type = input('Enter location type (City or ZIP):').lower()

        if self.location_type == self.CITY:
            self.city = input('Please provide City name:')
            rent_price, rent_forecast_12m, rent_forecast_60m = self.get_home_rent_price_by_city()
            sale_price, sale_forecast_12m, sale_forecast_60m = self.get_home_sale_price_by_city()
            print('Estimated price of you home is: %s. Rent price: %s' % (sale_price, rent_price))

            self.process_forecasts(rent_forecast_12m, rent_forecast_60m, sale_forecast_12m, sale_forecast_60m)

        elif self.location_type == self.ZIP:
            self.zip = input('Please provide ZIP Code:')
            rent_price, rent_forecast_12m, rent_forecast_60m = self.get_home_rent_price_by_zip()
            sale_price, sale_forecast_12m, sale_forecast_60m = self.get_home_sale_price_by_zip()
            print('Estimated price of you home is: %s. Rent price: %s' % (sale_price, rent_price))

            self.process_forecasts(rent_forecast_12m, rent_forecast_60m, sale_forecast_12m, sale_forecast_60m)

        else:
            input('Unknown location type "%s". Press Enter to exit' % self.location_type)
            exit()

    @staticmethod
    def process_forecasts(rent_forecast_12m, rent_forecast_60m, sale_forecast_12m, sale_forecast_60m):
        show_sale_forecast = input('Show price forecast for 12/60 months? (Y/n)')
        if show_sale_forecast.lower() in ('y', 'yes'):
            print('12 month price forecast: %s' % sale_forecast_12m)
            print('60 month price forecast: %s' % sale_forecast_60m)

        show_rent_forecast = input('Show rent forecast for 12/60 months? (Y/n)')
        if show_rent_forecast.lower() in ('y', 'yes'):
            print('12 month rent forecast: %s' % rent_forecast_12m)
            print('60 month rent forecast: %s' % rent_forecast_60m)

    def get_home_rent_price_by_city(self):
        """Shows estimated value for median rent for city"""
        datafile_name = 'data-files/City_MedianRentalPrice_%s' % self.BEDROOM_AMOUNT_FILE_NAMES[self.bedrooms_amount]
        file_data = pandas.read_csv(datafile_name, encoding='latin-1')
        file_data = file_data.set_index('RegionName')
        try:
            dataset = file_data.loc[self.city, file_data.columns[5:]]
        except KeyError:
            print('City "%s" not found in source data for your type of home. '
                  'Rent price cannot be predicted.' % self.city)
            return 'NaN'

        # calculating estimated rent price for next month (e.g. current) using exponental smoothing
        result = self.calculate_prediction(dataset.dropna())

        return result

    def get_home_rent_price_by_zip(self):
        """Shows estimated value for median rent for zip code"""
        datafile_name = 'data-files/Zip_MedianRentalPrice_%s' % self.BEDROOM_AMOUNT_FILE_NAMES[self.bedrooms_amount]
        file_data = pandas.read_csv(datafile_name, encoding='latin-1')
        file_data = file_data.set_index('RegionName')
        try:
            dataset = file_data.loc[self.zip, file_data.columns[6:]]
        except KeyError:
            print('Zip Code "%s" not found in source data for your type of home. '
                  'Rent price cannot be predicted.' % self.zip)
            return 'NaN'
        # calculating estimated rent price for next month (as last one is december of 2018)
        result = self.calculate_prediction(dataset.dropna())

        return result

    def get_home_sale_price_by_city(self):
        """Shows estimated value for a home based on median listing price for city"""
        datafile_name = 'data-files/City_MedianListingPrice_%s' % self.BEDROOM_AMOUNT_FILE_NAMES[self.bedrooms_amount]
        file_data = pandas.read_csv(datafile_name, encoding='latin-1')
        file_data = file_data.set_index('RegionName')
        try:
            dataset = file_data.loc[self.city, file_data.columns[5:]]
        except KeyError:
            print('City "%s" not found in source data for your type of home. '
                  'Home price cannot be predicted.' % self.city)
            return 'NaN'

        # calculating estimated rent price for next month (as last one is december of 2018)
        result = self.calculate_prediction(dataset.dropna())

        return result

    def get_home_sale_price_by_zip(self):
        """Shows estimated value for a home based on median listing price for zip code"""
        datafile_name = 'data-files/Zip_MedianListingPrice_%s' % self.BEDROOM_AMOUNT_FILE_NAMES[self.bedrooms_amount]
        file_data = pandas.read_csv(datafile_name, encoding='latin-1')
        file_data = file_data.set_index('RegionName')
        try:
            dataset = file_data.loc[self.zip, file_data.columns[6:]]
        except KeyError:
            print('Zip Code "%s" not found in source data for your type of home. '
                  'Home price cannot be predicted.' % self.zip)
            return 'NaN'

        # calculating estimated rent price for next month (as last one is december of 2018)
        result = self.calculate_prediction(dataset.dropna())

        return result

    @staticmethod
    def calculate_prediction(series, alpha=0.2):
        """
        Calculates next value based on previous ones, with exponential smoothed current series, using alpa var as weight,
        which makes older values weight less in new smoothed series
        """
        smoothed_series = [series[0]]  # first value is same as series
        for n in range(1, len(series)):
            smoothed_series.append(alpha * series[n] + (1 - alpha) * smoothed_series[n - 1])

        # Result for next moth is predicted using Brown's Simple Exponential Smoothing
        result = alpha*series[-1] + (1-alpha)*smoothed_series[-1]

        forecast_12m = HomePriceRentCalculator.predict_for_future(series, 12)
        forecast_60m = HomePriceRentCalculator.predict_for_future(series, 60)

        return result, forecast_12m, forecast_60m

    @staticmethod
    def predict_for_future(series, month_amount=12):
        """
        Predicting using Holt-Winters method and statsmodels library
        :return:
        """

        series.Timestamp = pandas.to_datetime(series.index, format='%Y-%m')
        series.index = series.Timestamp
        series.index.freq = 'MS'

        model = ExponentialSmoothing(np.asarray(series), seasonal='mul', trend='add', seasonal_periods=12).fit()
        forecast = model.forecast(month_amount)
        return forecast


HomePriceRentCalculator()