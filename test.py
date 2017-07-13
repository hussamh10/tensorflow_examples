import tensorflow as tf

# Weight Variable
W = tf.Variable([1], dtype=tf.float32)

#Bias Variable
b = tf.Variable([-1], dtype=tf.float32)

#placeholder inputs
x = tf.placeholder(tf.float32)

#perceptron
linear_model = W * x + b

#initializing session
sess = tf.Session()

#init variables
init = tf.global_variables_initializer()
sess.run(init)

#placeholder outputs
y = tf.placeholder(tf.float32)

# making a gradient descent perceptron
sqr_error = tf.square(linear_model - y)
loss = tf.reduce_sum(sqr_error)

#optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

#input and output
input = [1, 2, 3, 4]
output = [0, -1, -2, -3]

#training
for i in range(1000):
	sess.run(train, {x:input, y:output})


print(sess.run([W, b,]))

print(W, b)
print(sess.run(loss, {x:input, y:output}))
print(sess.run([W, b,]))
