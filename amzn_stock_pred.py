import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

from sklearn.preprocessing import MinMaxScaler
import math
import tensorflow as tf
from keras.layers import LSTM, Dense


df = pd.read_csv('AMZN_AAPL.csv')
# print(df.head(3))

# df['Difference'] = df['Close'] - df['Open']
# if df['Open'].any() < df['Close'].any():
#     df['Difference'] = '+' + df['Difference'].astype(str)
# elif df['Open'].any() > df['Close'].any():
#     df['Difference'] = '-' + df['Difference'].astype(str)
# print(df.head(3))

# # ## Add New column of "tickerName"
# df = df.assign(Ticker='AMZN')
## print(df.head(3))

# ## Print all Ticker names
all_stock_tick_names = df['Ticker'].unique()
# print(all_stock_tick_names)

li_split = [x.strip() for x in all_stock_tick_names.tolist()]
# print('After Split: ', li_split)

# ## Extracting for Ticker name
ticker = 'AAPL'  #input('Enter Ticker Name: ')
all_data = df['Ticker'] == ticker
final_data = df[all_data]
a = final_data.plot('Date', 'Close', color="red")
new_data = final_data.head(60)
b = new_data.plot('Date', 'Close', color="green")
# plt.show()

# ## Create new df and training set
close_data = final_data.filter(['Close'])
# print(close_data)
dataset = close_data.values
# print(dataset)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)
# print(scaled_data)
training_data_len = math.ceil(len(dataset) * 0.7)
train_data = scaled_data[0:training_data_len, :]
# print(len(dataset))
# print(training_data_len)
# print(train_data)

x_train = []
y_train = []
for i in range(60, len(train_data)):
    x_train = list(x_train)
    y_train = list(y_train)

    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

    x_train1, y_train1 = np.array(x_train), np.array(y_train)
    x_train2 = np.reshape(x_train1, (x_train1.shape[0], x_train1.shape[1], 1))


# ## LSTM Model
from keras.models import Sequential
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train2.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')
batch_size = int(len(x_train2) / 10)
epochs = 100
model.fit(x_train2, y_train1, batch_size=batch_size, epochs=epochs)

# ## Dataset for Testing
