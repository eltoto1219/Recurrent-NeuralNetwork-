#imports
from train_test_prep import *
import tensorflow as tf
from tensorflow.contrib import rnn

#train_y = train_y.reshape((216, 1))
#train_x = train_x.reshape((216,1,))
#test_y = test_y.reshape((36,1 ))
#test_x = test_x.reshape((36, 1))

#Other Params
n_epochs = 10
input_shape = 1
batch_size = 36
input_len = 216
n_neurons_l1 = 256
n_neurons_l2 = 128
n_neurons_l3 = 64
n_outputs =1

#PlaceHolders
x = tf.placeholder('float', [None, 1])
y = tf.placeholder('float', [None])

def deepModel(x):
        layer1 = {'weight':tf.Variable(tf.random_normal([1, n_neurons_l1])),
                  'biases':tf.Variable(tf.random_normal([n_neurons_l1]))}
        layer2 = {'weight':tf.Variable(tf.random_normal([n_neurons_l1, n_neurons_l2])),
                  'biases':tf.Variable(tf.random_normal([n_neurons_l2]))}
        layer3 = {'weight':tf.Variable(tf.random_normal([n_neurons_l2, n_neurons_l3])),
                  'biases':tf.Variable(tf.random_normal([n_neurons_l3]))}
        output = {'weight':tf.Variable(tf.random_normal([n_neurons_l3, n_outputs])),
                  'biases':tf.Variable(tf.random_normal([n_outputs]))}
        #input * weights + biases
        l1Comp = tf.add(tf.matmul(x, layer1['weight']), layer1['biases'])
        #activation function for layer 1 after (input * weights + biases) decides if neuron fires or not
        l1Comp = tf.nn.relu(l1Comp)

        l2Comp = tf.add(tf.matmul(l1Comp, layer2['weight']), layer2['biases'])
        l2Comp = tf.nn.relu(l2Comp)

        l3Comp = tf.add(tf.matmul(l2Comp, layer3['weight']), layer3['biases'])
        l3Comp = tf.nn.relu(l3Comp)

        outputComp = tf.add(tf.matmul(l3Comp, output['weight']), output['biases'])

        return outputComp

def trainModel(x):
    prediction = deepModel(x)
    #goal is to minimize cost, tf.reduce_mean = mean of tensor
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    #learning rate default is 0.001
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    # epoch is feedforward + backprop
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(n_epochs):
            #basicalling a placeholder until cost is caluclated
            cost = 0

            for _ in range(0, int((input_len / batch_size)-1)):
                start = int(_ * batch_size)
                end =  int(start+36)
                epoch_x = train_x[start: end]
                epoch_y = train_y[start: end]
                #How is tensoflow optimizing the cost?: by somehow updating the weights from our deepmodel
                _, c = sess.run([optimizer, cost], feed_dict = {x:epoch_x, y:epoch_y})
                #adds cost to variable epoch_cost, this variable resets for each epoch
                cost += c
            print('Epoch', epoch , 'completed of ', n_epochs, ". Cost: ", cost )

            pred = ses.run(outputComp, feed_dict={x: test_x})
            print(pred)

trainModel(x)



train_x[10:20]
