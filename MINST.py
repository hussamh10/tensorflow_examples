import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np

#A graph is a separate set of operations in tensor flow that is run outside of python

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# None means the dimension can be of any length
x = tf.placeholder(tf.float32, [None, 784])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.07).minimize(cross_entropy)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for _ in range(1000):
	bx, by = mnist.train.next_batch(100)
	sess.run(train_step, feed_dict={x:bx, y_:by})


prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))

acc = tf.reduce_mean(tf.cast(prediction, tf.float64))

print(sess.run(acc, feed_dict={x:mnist.test.images, y_:mnist.test.labels}))
