# Tensorflow Example:
# ---------------------------------------------------------------------------------------------
# This example will contruct a simple neural network to recognize handwriting digit (0, 1, ...9)
# ---------------------------------------------------------------------------------------------

# Import necessary modules---------------------------------------------------------------------
# After installing Tensorflow, it can be imported as a module
import tensorflow as tf
import input_data
# ---------------------------------------------------------------------------------------------

# ---------------------Load data into a tensor-------------------------------------------------
# Using input_data module to download images dataset and load in as ndarray (tensor)
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# ---------------------------------------------------------------------------------------------

# ---------------------Construction stage: A graph of computation is constructed---------------
# A simple model is described in form of the computation graph
# Placeholder or rank 2 tensor, of inputs: image of size 28x28 = 784 pixels, unlimited number of samples
x = tf.placeholder("float", [None, 784])
# Weights tensor rank 2, dxK where d is dimension of x, K is dimension of y
W = tf.Variable(tf.zeros([784,10]))
# Biases tensor rank 1, K dimensions
b = tf.Variable(tf.zeros([10]))
# Output is a softmax of linear function of input
y = tf.nn.softmax(tf.matmul(x,W) + b)
# Placeholder for placing real labels
y_ = tf.placeholder("float", [None,10])
# Cross entropy is a constraint for optimizing the weights
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
# Define the train step with GradientDescentOptimizer and minimized cross_entropy constraint
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
# All variables should be initialized explicitly = pushing this operation into the graph of computation
init = tf.initialize_all_variables()

# ---------------------Execution stage: Computation is conducted in a session------------------
# Model parameters are optimized in trend of minimizing cross_entropy
sess = tf.Session()
sess.run(init)
for i in range(1000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# ---------------------Evaluation stage: After training model, accuracy can be calculated------
predict_label = tf.argmax(y,1)
true_label = tf.argmax(y_,1)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
batch_xs = mnist.test.images
batch_ys = mnist.test.labels
result = sess.run([true_label, predict_label, correct_prediction], feed_dict={x: batch_xs, y_: batch_ys})
for i in range(1000):
	print 'True label ', result[0][i], '      ', 'predicted: ', result[1][i], '	', result[2][i]
print '--------------------------------------------'
print '----------------Accuracy',100 * sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}), '%------------------'

