Test project for predicting price for a home based on amount of bedrooms in specific area

Project is using Python 3.6

To launch:
- Create new virtualenvironment and activate
- pip install -r requirements.txt
- python calculate.py  # to calculate price
- python diabetes_prediction.py   # to calculate possibility of diabetes
- Put your values

Used https://people.duke.edu/~rnau/411avg.htm#SES as a model for predicitng closest result
Used Holt-Winter's method (Triple exponential smoothing) for prediciting future forecasts https://machinelearningmastery.com/exponential-smoothing-for-time-series-forecasting-in-python/

*Note:
In train.py file you can find method which was used to train the model with different params, and analyze the result.
Possibly python should be configured to use 'Tkinter' to work with method in train.py

