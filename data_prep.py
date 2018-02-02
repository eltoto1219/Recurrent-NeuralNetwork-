from dataFormat import *
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Scaling
scale = [i for i in range(n_rows)]
data = np.transpose([scale, prices])
scaler = MinMaxScaler()
scaler.fit(data)
data = scaler.transform(data)
data = np.transpose(data)
prices_scaled = data[1]
dates_scaled = data[0]
#code for un-scaling data
    #inversed = scaler.inverse_transform(np.transpose(data))
    #inversed = np.transpose(inversed)

#training data 0-216, testing data 217-252
train_y = prices_scaled[:216]
test_y = prices_scaled[216:]
train_x = dates_scaled[:216]
test_x= dates_scaled[216:]
dates_train = dates2[:216]
dates_test= dates2[216:]
dates_model = pd.DatetimeIndex(dates_test)
test_prices = prices[216:]

#reshaping the data
train_y = train_y.reshape((6, 1, 36))
train_x = train_x.reshape((6, 1, 36))
test_y = test_y.reshape((1,36))
test_x = test_x.reshape((1, 36))

if __name__ == "__main__":
    #the plot
    plt.figure(figsize=(10,5))
    plt.title('TSLA Stocks (Scaled Testing Data)')
    plt.ylabel('Scaled Price')
    plt.xlabel('Scaled Dates')
    plt.plot(test_x, test_y, '-b', label = 'Adjusted Closing Price')
    plt.legend(loc='upper right')
    plt.savefig('TSLA_Stocks_Scaled_Test.png')
