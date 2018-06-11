# Tensorflow Example:
# ---------------------------------------------------------------------------------------------
# This example will contruct a simple neural network to recognize handwriting digit (0, 1, ...9)
# ---------------------------------------------------------------------------------------------

# Import necessary modules---------------------------------------------------------------------
# After installing Tensorflow, it can be imported as a module
import tensorflow as tf
from sklearn import datasets
import numpy as np
# ---------------------------------------------------------------------------------------------

# ---------------------Load data into a tensor-------------------------------------------------
# Using digit datasets from sklearn
digits = datasets.load_digits()
n_samples = len(digits.images)
image_data = digits.images.reshape((n_samples, -1))  # Load image data and convert from 3D to 2D array
feature_size = image_data.shape[1]
label_size = 10
images_and_labels = list(zip(image_data, digits.target))
data_size = n_samples
# Define a batch size
batch_size = 10
if data_size % batch_size == 0:
    num_batches_per_epoch = data_size / batch_size
else:
    num_batches_per_epoch = int(data_size / batch_size) + 1


def convert_label(int_labels):
    hot_labels = []
    for label_ in int_labels:
        hot_label = [0.0 for _ in range(label_size)]
        hot_label[label_] = 1.0
        hot_labels.append(hot_label)
    return hot_labels


def batch_iter(_data, _batch_size, _batch_num):
    """
    :param _data
    :param _batch_size
    :param _batch_num
    Generates a batch iterator for a dataset.
    """
    _data = np.array(_data)
    np.random.shuffle(_data)
    shuffled_data = _data
    # Shuffle the data at each epoch
    start_index = _batch_num * _batch_size
    end_index = min((_batch_num + 1) * _batch_size, data_size)
    batch_data = shuffled_data[start_index:end_index]
    x_batch, y_batch = zip(*batch_data)
    return list(x_batch), list(y_batch)

# ---------------------------------------------------------------------------------------------

# ---------------------Construction stage: A graph of computation is constructed---------------
# A simple model is described in form of the computation graph
# Placeholder or rank 2 tensor, of inputs: image of size 28x28 = 784 pixels, unlimited number of samples
x = tf.placeholder("float", [None, feature_size])
# Weights tensor rank 2, dxK where d is dimension of x, K is dimension of y
W = tf.Variable(tf.zeros([feature_size, label_size]))
# Biases tensor rank 1, K dimensions
b = tf.Variable(tf.zeros([label_size]))
# Output is a softmax of linear function of input
y = tf.nn.softmax(tf.matmul(x, W) + b)
# Placeholder for placing real labels
y_ = tf.placeholder("float", [None, label_size])
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
print(len(images_and_labels))
print(images_and_labels[0])
for epoch in range(100):
    print('-' * 10 + str(epoch) + '-' * 10)
    for i in range(num_batches_per_epoch):
        batch_xs, batch_ys = batch_iter(images_and_labels, batch_size, i)
        batch_ys = convert_label(batch_ys)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# ---------------------Evaluation stage: After training model, accuracy can be calculated------
predict_label = tf.argmax(y, 1)
true_label = tf.argmax(y_, 1)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
test_size = 100
test_data = images_and_labels[:test_size]
batch_xs, batch_ys = zip(*test_data)
batch_xs = list(batch_xs)
batch_ys = list(batch_ys)
batch_ys = convert_label(batch_ys)
print(batch_xs[0])
print(batch_ys[0])

result = sess.run([true_label, predict_label, correct_prediction], feed_dict={x: batch_xs, y_: batch_ys})
for i in range(test_size):
    print 'True label ', result[0][i], '      ', 'predicted: ', result[1][i], '	', result[2][i]
print '--------------------------------------------'
print '----------------Accuracy', 100 * sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys}), '%------------------'
