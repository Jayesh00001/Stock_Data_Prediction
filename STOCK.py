import quandl as q
import pandas as pd
import numpy as np
import datetime
import yfinance as yf

from sklearn.linear_model import LinearRegression
# from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

df = q.get('WIKI/AMZN')

option = ['SPY', 'Snap Inc', 'AMZN', 'AAPL']
print(option)
name = 'AMZN'  # input('Stock Holder Name: ')

today = datetime.date.today()
# print(today)
before = today - datetime.timedelta(days=60)  # 3624


# df = yf.download(tickers=name, start=before, end='2023-02-06', progress=False)
# print(df.columns)

# df = pd.read_csv('AMZN.csv')
print(df.columns)
print(df.tail(3))

df = df[['Adj. Close']]
# print(df.tail(3))

forecast_out = int(30)
df['Prediction'] = df[['Adj. Close']].shift(-forecast_out)
# print(df.tail(3))
# df.to_csv('AAAA.csv')

x = np.array(df.drop(['Prediction'], axis=1))
x = preprocessing.scale(x)
print('Total Rows: ', len(x))

x_forecast = x[-forecast_out:]
x = x[: -forecast_out]
print(f'After {forecast_out}, x is: ', len(x))

y = np.array(df['Prediction'])
y = y[:-forecast_out]
print(f'After {forecast_out}, y is: ', len(y))

# print(df.tail(35))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(x.shape, x_train.shape, x_test.shape)

clf = LinearRegression()
clf.fit(x_train, y_train)
confidence = clf.score(x_test, y_test)
print("Confidence Score: ", confidence)

forecast_prediction = clf.predict(x_forecast)
print(forecast_prediction)

# df.to_csv('AAAA.csv')