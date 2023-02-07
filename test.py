# import pandas as pd

# df = pd.read_csv('ADANIENT_NS.csv')

# df['Difference'] = df['Close'] - df['Open']
# if df['Open'].any() < df['Close'].any():
#     df['Difference'] = '+' + df['Difference'].astype(str)
#     # print(df.head())

# elif df['Open'].any() > df['Close'].any():
#     df['Difference'] = '-' + df['Difference'].astype(str)
#     # print(df.head())

# # print(df.head())
# df = df[['Open', 'Close', 'Difference']]
# print(df.head())
# df.to_csv('hello.csv')



# df1 = pd.read_csv('AMZN.csv')
# df1 = df1.assign(Ticker='AMZN')
# # df1.to_csv('ADANIENT_NS_train.csv', index=False)
# # print(df1.head())
# df2 = pd.read_csv('AAPL.csv')
# df2 = df2.assign(Ticker='AAPL')
# df_concat = pd.concat([df1, df2])
# df_concat.to_csv('concatenated_dataframe.csv', index=False)





############################################


# import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from keras.layers import LSTM, Dense

# load data
data = pd.read_csv('AMZN.csv')

# extract input and output data
X = data[['Open', 'High', 'Low', 'Volume']].values
y = data[['Close']].values

# scale data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
y_scaled = scaler.fit_transform(y)

# define model
from keras.models import Sequential
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_scaled.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')
batch_size = int(len(X_scaled) / 10)
epochs = 50
model.fit(X, y, batch_size=batch_size, epochs=epochs)

# # compile model
# model.compile(loss='mse', optimizer='adam')

# # fit model
# model.fit(X_scaled, y_scaled, epochs=200, batch_size=10, verbose=2)

# predict value
input_data = np.array([[110.25, 114.0, 108.879997, 156164800]])
input_data_scaled = scaler.fit_transform(input_data)
next_date_close_value = model.predict(input_data_scaled)[0]

# denormalize the prediction
next_date_close_value = scaler.inverse_transform(next_date_close_value.reshape(-1,1))[0][0]

print("The predicted Close value for the next date is: ", next_date_close_value)
