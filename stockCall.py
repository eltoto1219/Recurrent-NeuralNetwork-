#using yahoo api to get stock data and convert to .csv

#imports
import pandas as pd
import pandas_datareader.data as web
import datetime as dt

#setting start and end dates for the data we wanna call
start = dt.datetime(2016, 1, 1)
end = dt.datetime(2016, 12, 31)

#Calling in the data
df =  web.DataReader('TSLA', 'yahoo', start, end)

#Writing to csv
df.to_csv('TSLA.csv')
