import tensorflow as tf
import numpy as np
from data_loader import FileDAO, FeatureGen
from model import MLP


def batch_iter(_data, _batch_size, _batch_num):
    """
    :param _data
    :param _batch_size
    :param _batch_num
    Generates a batch iterator for a dataset.
    """
    data_size = len(_data)
    start_index = _batch_num * _batch_size
    end_index = min((_batch_num + 1) * _batch_size, data_size)
    batch_data = _data[start_index:end_index]
    return batch_data


def balance_data(data, labels):
    b_data_0 = []
    b_label_0 = []
    b_data_1 = []
    b_label_1 = []
    for id_, label_arr in enumerate(labels):
        if label_arr.index(max(label_arr)) == 0:
            b_data_0.append(data[id_])
            b_label_0.append(label_arr)
        else:
            b_data_1.append(data[id_])
            b_label_1.append(label_arr)
    min_size = min([len(b_data_0), len(b_data_1)])
    reduce_size = min_size
    b_data = b_data_0[:min_size] + b_data_1[:reduce_size]
    b_label = b_label_0[:min_size] + b_label_1[:reduce_size]
    print(len(b_data_0))
    print(len(b_data_1))
    return [b_data, b_label]


def train():
    checkpoint_prefix = 'models/model'
    file_dao = FileDAO("data/lottery_data")
    data = file_dao.get_data()
    feature_gen = FeatureGen(data)
    data, labels = feature_gen.get_train_data(_index_range=20, _feature_size=10, _offset=0)
    data, labels = balance_data(data, labels)
    total_data_size = len(data)
    shuffle_indices = np.random.permutation(np.arange(total_data_size))
    shuffle_data = np.array(data)[shuffle_indices]
    shuffle_labels = np.array(labels)[shuffle_indices]
    split_index = int(0.9 * total_data_size)
    train_data = shuffle_data[:split_index]
    train_labels = shuffle_labels[:split_index]
    np_train_data = np.array(train_data)
    np_train_labels = np.array(train_labels)
    test_data = shuffle_data[split_index:]
    test_labels = shuffle_labels[split_index:]

    batch_size = 3
    data_size = len(train_data)
    if data_size % batch_size == 0:
        num_batches_per_epoch = data_size / batch_size
    else:
        num_batches_per_epoch = int(data_size / batch_size) + 1

    feature_size = len(data[0])
    label_size = len(labels[0])
    hidden_size = 30
    mlp = MLP(feature_size=feature_size, hidden_size=hidden_size, label_size=label_size)
    init = tf.initialize_all_variables()
    sess = tf.Session()
    writer = tf.train.SummaryWriter("/tmp/lottery_logs", sess.graph_def)
    saver = tf.train.Saver(tf.all_variables(), max_to_keep=1)
    sess.run(init)
    for epoch in range(200):
        print('-' * 10 + 'epoch: ' + str(epoch) + '-' * 10)
        if epoch % 10 == 0:
            batch_xs = test_data
            batch_ys = test_labels
            feed = {mlp.x: batch_xs, mlp.y_: batch_ys}
            result = sess.run([mlp.merged, mlp.accuracy], feed_dict=feed)
            summary_str = result[0]
            acc = result[1]
            writer.add_summary(summary_str, epoch)
            print("Accuracy at step %s: %s" % (epoch, acc))
            saver.save(sess, checkpoint_prefix, global_step=epoch)

        shuffle_indices = np.random.permutation(np.arange(len(train_data)))
        shuffle_data = np_train_data[shuffle_indices]
        shuffle_labels = np_train_labels[shuffle_indices]

        for i in range(num_batches_per_epoch):
            batch_xs = batch_iter(shuffle_data, batch_size, i)
            batch_ys = batch_iter(shuffle_labels, batch_size, i)
            sess.run(mlp.train_step, feed_dict={mlp.x: batch_xs, mlp.y_: batch_ys})

            # ---------------------Evaluation stage: After training model, accuracy can be calculated------

    batch_xs = test_data
    batch_ys = test_labels
    test_size = len(test_data)
    result = sess.run([mlp.true_label, mlp.predict_label, mlp.correct_prediction, mlp.score],
                      feed_dict={mlp.x: batch_xs, mlp.y_: batch_ys})
    for i in range(test_size):
        print 'True label ', result[0][i], '      ', 'predicted: ', result[1][i], '	', result[2][i], ' prob: ', result[3][i]
    print '--------------------------------------------'
    print '----------------Accuracy', 100 * sess.run(mlp.accuracy, feed_dict={mlp.x: batch_xs, mlp.y_: batch_ys}), '%------------------'

    # To run tensorboard we use:
    # python tensorflow/tensorboard/tensorboard.py --logdir=/tmp/lottery_logs
    # If we installed tf with pip, tensorboard is installed into the system path. We can use:
    # tensorboard --logdir=/tmp/lottery_logs


def main():
    print('start')
    train()


if __name__ == '__main__':
    main()

