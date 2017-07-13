import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np

def LinearRegression(features, labels, mode):
	W = tf.get_variable("W", [1], dtype=tf.float64)
	b = tf.get_variable("b", [1], dtype=tf.float64)
	y = W*features['x'] + b

	loss = tf.reduce_sum(tf.square(y - labels))
	global_step = tf.train.get_global_step()
	optimizer = tf.train.GradientDescentOptimizer(0.01)

	train = tf.group(optimizer.minimize(loss), tf.assign_add(global_step, 1))

	return tf.contrib.learn.ModelFnOps(
		mode=mode, predictions=y,
		loss=loss,
		train_op=train
	)


#Declaring a single list of features "x"
features = [tf.contrib.layers.real_valued_column("x", dimension=1)]
print(features)

#Making a model for estimating
estimator = tf.contrib.learn.Estimator(model_fn=LinearRegression)

#setting up data for training and testing
x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])

x_test = np.array([2., 5., 8., 1.])
y_test = np.array([-1.01, -4.1, -7., 0.])

input_fn = tf.contrib.learn.io.numpy_input_fn({"x":x_train}, y_train, batch_size=4, num_epochs=1000)
eval_input_fn = tf.contrib.learn.io.numpy_input_fn({"x":x_train}, y_train, batch_size=4, num_epochs=1000)

# train the estimator on input for 1000 steps
estimator.fit(input_fn=input_fn, steps=1000)

#now evaluate the loss of training data
train_loss = estimator.evaluate(input_fn=input_fn)

#test loss
test_loss = estimator.evaluate(input_fn=eval_input_fn)

print(train_loss, test_loss)


