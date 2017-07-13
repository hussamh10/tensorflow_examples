import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

sess = tf.InteractiveSession()

# the x and y_ nodes of the graphs are intialized as the placeholders
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

sess.run(tf.global_variables_initializer())

# the basic shape of our graph y
y = tf.matmul(x, W) + b

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

for _ in range(1000):
	batch = mnist.train.next_batch(100)
	sess.run(train_step, feed_dict={x:batch[0], y_:batch[1]})

correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
a = tf.cast(correct_prediction, tf.float32)
print(sess.run([correct_prediction, a], feed_dict={x:mnist.test.images, y_:mnist.test.labels}))

print(accuracy.eval(feed_dict={x:mnist.test.images, y_:mnist.test.labels}))


