import pandas as pd
import numpy as np
from  datetime import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates

#Read data
data = pd.read_csv('TSLA.csv', parse_dates=True, index_col=0)
prices  = data['Adj Close']
n_rows = prices.shape[0]
n_columns = 1
prices = prices.values
dates = data.index
dates2 = dates.values


if __name__ == "__main__":
    #the plot
    plt.figure(figsize=(10,5))
    plt.title('TSLA Stocks')
    plt.ylabel('Price: $')
    plt.xlabel('Dates')
    plt.plot(dates, prices, '-b', label = 'Adjusted Closing Price')
    #plt.plot(dates, prices2, '-r', label= 'Bye')
    plt.legend(loc='upper right')
    plt.savefig('TSLA_Stocks.png')
    #plt.scatter(prices, prices2)
    #plt.show()
