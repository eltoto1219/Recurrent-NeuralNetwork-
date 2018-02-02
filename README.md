# Recurrent-NeuralNetwork-

The goal of this model is to experiment with how well a basic Recurrent Neural Network with Long-Short-Term-Memory cells can predict the
daily adjusted daily closing prices for the company Tesla in December of 2016. First,
I retrieved a full year of historical financial data using the Yahoo finance api. Next,
I chose which features I would like to use in my model to make a price prediction.
Then, I scaled and formatted my data to feed through a Recurrent Neural Network that
I programmed using the TensorFlow library in python. Finally, I plotted the network’s
predictions and other important indicators to gauge how well my network performed
with the financial data I trained it with. Ultimately, my RNN accurately predicted the
overall price-direction for a 36 day time period and roughly modeled the daily adjusted
close prices to a point where there were distinct similarities between the predictions
and actual prices. However, this model was crude at predicting daily price directions."