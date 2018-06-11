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
hidden_size = 20
label_size = 10
total_images_and_labels = np.array(list(zip(image_data, digits.target)))

# Split into train and test data
shuffle_indices = np.random.permutation(np.arange(n_samples))
shuffle_data = total_images_and_labels[shuffle_indices]
data_size = int(0.9 * n_samples)

images_and_labels = shuffle_data[:data_size]
test_data = shuffle_data[data_size:]


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
    start_index = _batch_num * _batch_size
    end_index = min((_batch_num + 1) * _batch_size, data_size)
    batch_data = _data[start_index:end_index]
    x_batch, y_batch = zip(*batch_data)
    return list(x_batch), list(y_batch)

# ---------------------------------------------------------------------------------------------

# ---------------------Construction stage: A graph of computation is constructed---------------
# A simple model is described in form of the computation graph
# Placeholder or rank 2 tensor, of inputs: image of size 28x28 = 784 pixels, unlimited number of samples
x = tf.placeholder("float", [None, feature_size])
# Hidden layer
Wh = tf.Variable(tf.random_uniform([feature_size, hidden_size], -1.0, 1.0))
bh = tf.Variable(tf.random_uniform([hidden_size], -1.0, 1.0))
h = tf.sigmoid(tf.matmul(x, Wh) + bh)
# Weights tensor rank 2, dxK where d is dimension of x, K is dimension of y
# W = tf.Variable(tf.zeros([feature_size, label_size]))
W = tf.Variable(tf.random_uniform([hidden_size, label_size], -1.0, 1.0))
# Biases tensor rank 1, K dimensions
# b = tf.Variable(tf.zeros([label_size]))
b = tf.Variable(tf.random_uniform([label_size], -1.0, 1.0))
# Output is a softmax of linear function of input
y = tf.nn.softmax(tf.matmul(h, W) + b)
# Placeholder for placing real labels
y_ = tf.placeholder("float", [None, label_size])
# Cross entropy is a constraint for optimizing the weights
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
ce_summ = tf.summary.scalar("cross entropy", cross_entropy)
# Define the train step with GradientDescentOptimizer and minimized cross_entropy constraint
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
# All variables should be initialized explicitly = pushing this operation into the graph of computation

# For evaluation
predict_label = tf.argmax(y, 1)
true_label = tf.argmax(y_, 1)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
accuracy_summary = tf.summary.scalar("accuracy", accuracy)

# Merge all the summaries
merged = tf.summary.merge_all()

# Initialize all variables
init = tf.initialize_all_variables()

# ---------------------Execution stage: Computation is conducted in a session------------------
# Model parameters are optimized in trend of minimizing cross_entropy
sess = tf.Session()

# Write summaries out to /tmp/mnist_logs
writer = tf.summary.FileWriter("/tmp/mnist_logs", sess.graph_def)

sess.run(init)
for epoch in range(100):
    print('-' * 10 + str(epoch) + '-' * 10)
    # if epoch % 10 = 0, collect accuracy for tensorboard
    if epoch % 10 == 0:
        batch_xs, batch_ys = zip(*test_data)
        batch_xs = list(batch_xs)
        batch_ys = list(batch_ys)
        batch_ys = convert_label(batch_ys)
        feed = {x: batch_xs, y_: batch_ys}
        result = sess.run([merged, accuracy], feed_dict=feed)
        summary_str = result[0]
        acc = result[1]
        writer.add_summary(summary_str, epoch)
        print("Accuracy at step %s: %s" % (epoch, acc))

    # Shuffle the data at each epoch
    shuffle_indices = np.random.permutation(np.arange(data_size))
    shuffle_data = images_and_labels[shuffle_indices]
    for i in range(num_batches_per_epoch):
        batch_xs, batch_ys = batch_iter(shuffle_data, batch_size, i)
        batch_ys = convert_label(batch_ys)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        # test_y = sess.run(y, feed_dict={x: batch_xs})
        # print(test_y[0])
    # batch_xs, batch_ys = zip(*images_and_labels)
    # batch_xs = list(batch_xs)
    # batch_ys = list(batch_ys)
    # batch_ys = convert_label(batch_ys)
    # sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

writer.flush()
writer.close()

# ---------------------Evaluation stage: After training model, accuracy can be calculated------

test_size = len(test_data)
# test_data = images_and_labels[:test_size]
batch_xs, batch_ys = zip(*test_data)
batch_xs = list(batch_xs)
batch_ys = list(batch_ys)
batch_ys = convert_label(batch_ys)

result = sess.run([true_label, predict_label, correct_prediction], feed_dict={x: batch_xs, y_: batch_ys})
for i in range(test_size):
    print 'True label ', result[0][i], '      ', 'predicted: ', result[1][i], '	', result[2][i]
print '--------------------------------------------'
print '----------------Accuracy', 100 * sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys}), '%------------------'

# To run tensorboard we use:
# python tensorflow/tensorboard/tensorboard.py --logdir=/tmp/mnist_logs
# If we installed tf with pip, tensorboard is installed into the system path. We can use:
# tensorboard --logdir=/tmp/mnist_logs
# Finally go to browser and open address: http://localhost:6006
