import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import os

# ##### NOTICE #####
# Please download at: http://files.statworx.com/sp500.zip


def load_data(file_path):
    # Import data
    data = pd.read_csv(file_path)
    # Drop date variable
    data = data.drop(['DATE'], 1)
    # Dimensions of dataset
    n = data.shape[0]
    p = data.shape[1]
    # Make data a numpy array
    data = data.values
    return data, n, p


def split_data(data, n, p):
    # Training and test data
    train_start = 0
    train_end = int(np.floor(0.8 * n))
    test_start = train_end + 1
    test_end = n
    data_train = data[np.arange(train_start, train_end), :]
    data_test = data[np.arange(test_start, test_end), :]
    return data_train, data_test


def scale_data(data_train, data_test):
    # Scale data
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(data_train)
    data_train = scaler.transform(data_train)
    data_test = scaler.transform(data_test)
    # Build X and y
    X_train = data_train[:, 1:]
    y_train = data_train[:, 0]
    X_test = data_test[:, 1:]
    y_test = data_test[:, 0]
    return X_train, y_train, X_test, y_test


def process_data(input_path='data_stocks.csv'):
    data, n, p = load_data(input_path)
    data_train, data_test = split_data(data, n, p)
    X_train, y_train, X_test, y_test = scale_data(data_train, data_test)
    return X_train, y_train, X_test, y_test


def build_model():
    # Model architecture parameters
    n_stocks = 500
    n_neurons_1 = 1024
    n_neurons_2 = 512
    n_neurons_3 = 256
    n_neurons_4 = 128
    n_target = 1

    # Placeholder
    X = tf.placeholder(dtype=tf.float32, shape=[None, n_stocks])
    Y = tf.placeholder(dtype=tf.float32, shape=[None])

    # Initializers
    sigma = 1
    weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
    bias_initializer = tf.zeros_initializer()

    # Layer 1: Variables for hidden weights and biases
    W_hidden_1 = tf.Variable(weight_initializer([n_stocks, n_neurons_1]))
    bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))
    # Layer 2: Variables for hidden weights and biases
    W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
    bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))
    # Layer 3: Variables for hidden weights and biases
    W_hidden_3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
    bias_hidden_3 = tf.Variable(bias_initializer([n_neurons_3]))
    # Layer 4: Variables for hidden weights and biases
    W_hidden_4 = tf.Variable(weight_initializer([n_neurons_3, n_neurons_4]))
    bias_hidden_4 = tf.Variable(bias_initializer([n_neurons_4]))

    # Output layer: Variables for output weights and biases
    W_out = tf.Variable(weight_initializer([n_neurons_4, n_target]))
    bias_out = tf.Variable(bias_initializer([n_target]))

    # Hidden layer
    hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))
    hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))
    hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))
    hidden_4 = tf.nn.relu(tf.add(tf.matmul(hidden_3, W_hidden_4), bias_hidden_4))

    # Output layer (must be transposed)
    out = tf.transpose(tf.add(tf.matmul(hidden_4, W_out), bias_out))

    # Cost function
    mse = tf.reduce_mean(tf.squared_difference(out, Y))
    # Optimizer
    opt = tf.train.AdamOptimizer().minimize(mse)
    return X, Y, opt, out, mse, [W_hidden_1, bias_hidden_1, W_hidden_2,
                                 bias_hidden_2, W_hidden_3, bias_hidden_3,
                                 W_hidden_4, bias_hidden_4, W_out, bias_out]


def train(X_train, y_train, X_test, y_test, X, Y, opt, out, mse, output_model='outputs'):
    if not os.path.exists(output_model):
        os.makedirs(output_model)
    # Make Session
    net = tf.Session()
    # Run initializer
    net.run(tf.global_variables_initializer())

    # Setup interactive plot
    plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    line1, = ax1.plot(y_test)
    line2, = ax1.plot(y_test * 0.5)
    plt.show()

    # Number of epochs and batch size
    epochs = 10
    batch_size = 256

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    for e in range(epochs):

        # Shuffle training data
        shuffle_indices = np.random.permutation(np.arange(len(y_train)))
        X_train = X_train[shuffle_indices]
        y_train = y_train[shuffle_indices]

        # Minibatch training
        for i in range(0, len(y_train) // batch_size):
            start = i * batch_size
            batch_x = X_train[start:start + batch_size]
            batch_y = y_train[start:start + batch_size]
            # Run optimizer with batch
            net.run(opt, feed_dict={X: batch_x, Y: batch_y})

            # Show progress
            if np.mod(i, 5) == 0:
                # Prediction
                pred = net.run(out, feed_dict={X: X_test})
                line2.set_ydata(pred)
                plt.title('Epoch ' + str(e) + ', Batch ' + str(i))
                file_name = 'images/epoch_' + str(e) + '_batch_' + str(i) + '.jpg'
                plt.savefig(file_name)
                plt.pause(0.01)
            if np.mod(i, 20) == 0:
                save_path = saver.save(net, os.path.join(output_model, "model.ckpt"))
                print("Model saved in path: %s" % save_path)
    # Print final MSE after Training
    mse_final = net.run(mse, feed_dict={X: X_test, Y: y_test})
    print(mse_final)


def test(variables, X, Y, out, mse, X_test, y_test, output_model='outputs'):
    # Add ops to save and restore only `v2` using the name "v2"
    saver = tf.train.Saver(variables)

    # Use the saver object normally after that.
    with tf.Session() as sess:
        saver.restore(sess, os.path.join(output_model, "model.ckpt"))
        preds, loss_mse = sess.run([out, mse], feed_dict={X: X_test, Y: y_test})
        print('prediction {}'.format(preds))
        print('mean square error {}'.format(loss_mse))


def train_job():
    X_train, y_train, X_test, y_test = process_data()
    X, Y, opt, out, mse, _ = build_model()
    train(X_train, y_train, X_test, y_test, X, Y, opt, out, mse)


def test_job():
    _, _, X_test, y_test = process_data()
    X, Y, opt, out, mse, variables = build_model()
    test(variables, X, Y, out, mse, X_test, y_test)

if __name__ == '__main__':
    # train_job()
    test_job()
