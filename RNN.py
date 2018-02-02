#imports
from data_prep import *
import tensorflow as tf
from tensorflow.contrib import rnn
#Reshaping the data-- input: (6, 36) labels: (6, 36)
n_epochs = 50
rnn_size = 198
n_seqs = 6
timestep =6
batch_size = 6
n_outputs = 36

#PlaceHolders
x = tf.placeholder('float', shape=[None,  n_seqs])
y = tf.placeholder('float')


def computational_graph(x):

    layer = {'weights':tf.Variable(tf.random_normal([rnn_size, n_outputs])),
             'biases' :tf.Variable(tf.random_normal([n_outputs]))}

    x = tf.reshape(x, [-1, n_seqs])
    x = tf.split(x, timestep, 0)

    lstm_cell = rnn.BasicLSTMCell(rnn_size)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    output = tf.add(tf.matmul(outputs[-1], layer['weights']), layer['biases'])
    print(output)

    return output


def train_graph(x):
    output = computational_graph(x)

    costFunc = tf.reduce_mean( tf.squared_difference(output, y))
    optFunc = tf.train.AdamOptimizer(learning_rate=0.003).minimize(costFunc)

    #starting tf session
    with tf.Session() as ses:
        #intitializing all tf variables
        ses.run(tf.global_variables_initializer())
        #lists for plotting the loss and price action
        loss_list = []
        accuracies = []

        for epoch in range(n_epochs):

                #batch training
                for i in range(0, batch_size-1):
                    #total loss is added conseuctively per epoch
                    epoch_loss = 0
                    batch_x = train_x.reshape((batch_size, n_seqs, timestep))
                    batch_y = train_y
                    start = i * batch_size
                    batch_x = batch_x[i]
                    batch_y = batch_y[i]
                    # Run optimizer for each batch
                    i, c = ses.run([optFunc, costFunc], feed_dict={x: batch_x, y: batch_y})
                    epoch_loss += c
                    loss_list.append(epoch_loss)


                    print('Epoch-Loss-', str(epoch), '-of-',str(n_epochs),'-',str(epoch_loss))

                    # retreiving the predictions
                    pred = ses.run(output, feed_dict={x: test_x.reshape((n_seqs, timestep))})
                    pred = pred.tolist()
                    #print(pred)

                    #un-scaling predicted data into real data
                    plot_x = test_x.tolist()
                    holder1 = []
                    holder2 = []
                    for w in pred:
                        for z in w:
                            holder1.append(z)
                    for w in plot_x:
                        for z in w:
                            holder2.append(z)
                    plot_data = np.transpose([holder2, holder1])
                    inversed = scaler.inverse_transform(plot_data)
                    inversed = np.transpose(inversed)
                    inversed = list(reversed(inversed[1]))

                    #making the  stock figure
                    plt.figure(figsize=(10,5))
                    plt.title('TSLA Stocks Prediction')
                    plt.ylabel('Price: $')
                    plt.xlabel('Dates')
                    plt.plot(dates_model, test_prices, '-b', label = 'Adjusted Closing Price')
                    plt.plot(dates_model, inversed, '-r', label = 'Adjusted Closing Price Prediction')
                    plt.legend(loc='upper right')
                    plt.grid()
                    file_name = 'epoch_graph2/epoch_' + str(epoch) +  '.jpg'
                    plt.savefig(file_name)
                    plt.clf()

                    #making scatterplot
                    plt.figure(figsize=(5,5))
                    plt.title('TSLA Predicted vs. Actual Stocks')
                    plt.ylabel('Predicted Price')
                    plt.xlabel('Actual Price')
                    plt.scatter(test_prices, inversed, label = 'Adjusted Closing Price')
                    plt.grid()
                    file_name2 = 'epoch_graph2/Scatter_epoch_' + str(epoch) +  '.jpg'
                    plt.savefig(file_name2)
                    plt.clf()

                #making price-action accuracy function

                a = test_prices
                b = inversed
                e = [i for i in range(1, 51)]
                accuracy = 0
                for i in range(0, (len(a)-1)):

                    if (a[i] - a[i+1] < 0 and b[i] - b[i+1] < 0) or (a[i] - a[i+1] > 0 and b[i] - b[i+1] > 0):
                            accuracy += 1

                z = 100 * accuracy/(len(a)-1)
                accuracies.append(z)
                print('Epcoh', epoch, 'Accuracy: ', z, '%')

        #making error plot
        current_epoch = [((i*(1/(batch_size-1)))) for i in range(0,int(((batch_size-1)*n_epochs)))]
        plt.figure(figsize=(5,5))
        plt.title('Mean Squared Error')
        plt.ylabel('Error Value')
        plt.xlabel('Epoch number')
        plt.plot(current_epoch, loss_list)
        plt.grid()
        file_name3 = 'errorGraph.jpg'
        plt.savefig(file_name3)
        plt.clf()

        #price-wise accruacy plot
        plt.figure(figsize=(5,5))
        plt.title('Price Action Accuracy')
        plt.ylabel('Percentage Correct %')
        plt.xlabel('Epoch number')
        plt.bar(e, accuracies, width=0.5, color='green', edgecolor='black')
        plt.yticks([0,10,20,30,40,50,60,70,80,90,100])
        plt.grid()
        file_name4 = 'accuracyGraph.jpg'
        plt.savefig(file_name4)
        plt.clf()




train_graph(x)
