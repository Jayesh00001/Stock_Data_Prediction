
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from ta.volatility import BollingerBands
from ta.trend import MACD
from ta.momentum import RSIIndicator
import datetime
import pandas_ta as ta

st.markdown("<h2 style='text-align: center; color: green;'>ALGO TRADING</h2>", unsafe_allow_html=True)

# df = pd.read_csv('ADANIENT_NS.csv')
# st.write('Total features: ', len(df.columns))
# st.dataframe(df)

option = st.sidebar.selectbox('Select Here: ', ('SPY', 'SNAP', 'AMZN', 'AAPL', 'BBBY', 'PINS', 'AMC', 'ATVI', 'CTLT', 'NCMGF', 'NCMGY', 'LSI', 'VERX'))
# name = st.write('Stock Holder Name: ', option)


###########
# SideBar #
###########

today = datetime.date.today()
before = today - datetime.timedelta(days=37)  # 3624
# st.write(before)

start_date = st.sidebar.date_input('Start Date: ', before)
# st.write('Start Date: ', start_date)

end_date = st.sidebar.date_input('End Date: ', today)
# st.write('End Date: ', end_date)

col1, col2 = st.columns(2)
with col1:
    st.markdown(f"<h5 style='text-align: center; color: black;'>Stock Holder Name: {option}</h5>", unsafe_allow_html=True)

with col2:
    days = end_date - start_date
    st.markdown(f"<h5 style='text-align: center; color: black;'>Total Selected Days: {days}</h5>", unsafe_allow_html=True)

st.write('')
st.write('')

if start_date < end_date:
    st.sidebar.success('Start Date: "%s" \n\nEnd Date: "%s"' % (start_date, end_date))
else:
    st.sidebar.error('Error: The end date must be after the start date.')


##############
# Stock Data #
##############

df = yf.download(tickers=option, start=start_date, end=end_date, progress=False)
# st.write('COLUMNS OF TABLE: ', df.columns)


df['Difference'] = df['Close'] - df['Open']
if df['Open'].any() < df['Close'].any():
    df['Difference'] = '+' + df['Difference'].astype(str)

elif df['Open'].any() > df['Close'].any():
    df['Difference'] = '-' + df['Difference'].astype(str)


st.write('Actual Dataset: ', df.head())
# df = df[['Open', 'Close', 'Difference']]

indicator_bb = BollingerBands(df['Close'])
# st.write(indicator_bb)
bb = df
bb['bb_h'] = indicator_bb.bollinger_hband()
bb['bb_l'] = indicator_bb.bollinger_lband()
bb['bb_h'] = bb['Open'] + bb['Close']
bb = bb[['Close', 'bb_h', 'bb_l']]
# st.write(bb.tail(30))

# macd1 = MACD(df['Close']).macd()
# # macd2 = ta.macd(bb['Close'])
# st.write('MACD TABLE: ', macd1)

# rsi = RSIIndicator(df['Close']).rsi()
# st.write('RSI TABLE: ', rsi)

col1, col2 = st.columns(2)
with col1:
    macd1 = MACD(df['Close']).macd()
    # macd2 = ta.macd(bb['Close'])
    st.write('MACD TABLE: ', macd1)
with col2:
    rsi = RSIIndicator(df['Close']).rsi()
    st.write('RSI TABLE: ', rsi)



#############
# Main Page #
#############

# ## Plot the prices and the bollinger bands
st.write('Stock Bollinger Bands: ')
st.line_chart(bb)

# progress_bar = st.progress(10)

# ##Plot MACD
st.write('Stock Moving Average Convergence Divergence (MACD): ')
st.area_chart(macd1)

# ## Plot RSI
st.write('Stock RSI: ')
st.line_chart(rsi)

# ## Data of recent days
st.write('Recent 10 Days Data: ')
st.dataframe(df.tail(10))

# col1, col2 = st.columns(2)
# with col1:
#    # ## Plot the prices and the bollinger bands
#     st.write('Stock Bollinger Bands: ')
#     st.line_chart(bb)
# with col2:
#    # ##Plot MACD
#     st.write('Stock Moving Average Convergence Divergence (MACD): ')
#     st.area_chart(macd1)

# with col1:
#     # ## Plot RSI
#     st.write('Stock RSI: ')
#     st.line_chart(rsi)
# with col2:
#     # ## Data of recent days
#     st.write('Recent Data: ')
#     st.dataframe(df.tail(10))







# import base64
# from io import BytesIO

# def to_excel(df):
#     output = BytesIO()
#     writer = pd.ExcelWriter(output, engine='xlsxwriter')
#     df.to_excel(writer, sheet_name='Sheet1')
#     writer.save()
#     processed_data = output.getvalue() 
#     return processed_data

# def get_table_download_link(df):
#     """
#     Generates a link allowing the data in a given panda dataframe to be downloaded
#     in:  dataframe
#     out: href string
#     """
#     val = to_excel(df)
#     b64 = base64.b64encode(val)  # val looks like b'...'
#     return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="download.xlsx">Download excel file</a>' # decode b'abc' => abc

# st.markdown(get_table_download_link(df), unsafe_allow_html=True)
