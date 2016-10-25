import tensorflow as tf
import pickle
import numpy as np


NUM_EPOCHS = 100000

rawX = pickle.load(open('aligned_tiles.pickle','rb'))
rawY = pickle.load(open('aligned_targets.pickle','rb'))

dataX  = np.vstack([x[180:220, 180:220, :].flatten() for x in rawX])
dataY = np.vstack([y[180:220, 180:220] for y in rawY]).reshape((101, 1600))

trX = dataX[:80] 
trY = dataY[:80]

valX = dataX[80:]
valY = dataY[80:]


X = tf.placeholder("float", [None, 4800])
Y = tf.placeholder("float", [None, 1600])

W_h = tf.Variable(tf.random_normal([4800, 10], stddev=0.00001))
b_h = tf.Variable(tf.zeros([1,10]))

W_o = tf.Variable(tf.random_normal([10,1600], stddev=0.00001))
#b_o = tf.Variable(tf.zeros([1,1]))

H = tf.nn.relu(tf.matmul(X, W_h) + b_h)

Yhat = tf.matmul(H, W_o) # + b_o

cost = tf.reduce_mean(tf.square(tf.sub(Yhat, Y)))

train_op = tf.train.GradientDescentOptimizer(.00001).minimize(cost)


predictions = []


with tf.Session() as sess:

  tf.initialize_all_variables().run()

  for e in range(NUM_EPOCHS):
    sess.run(train_op, feed_dict={X: trX, Y: trY})
    epoch_cost = sess.run(cost, feed_dict={X: trX, Y: trY})
    valid_cost = sess.run(cost, feed_dict={X: valX, Y: valY})
    print("Cost after epoch ", e, ": ", epoch_cost, " Valid cost: ", valid_cost)
    predictions = sess.run(Yhat, feed_dict={X: trX}) 

  #for x in trX:
  #  predictions.append(sess.run(Yhat, feed_dict={X: [x]}))
















