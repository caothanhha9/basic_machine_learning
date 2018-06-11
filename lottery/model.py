import tensorflow as tf


class MLP(object):
    def __init__(self, feature_size, hidden_size, label_size):
        self.hidden_size = hidden_size
        self.feature_size = feature_size
        self.label_size = label_size
        self.cross_entropy = None
        self.ce_summ = None
        self.train_step = None
        self.predict_label = None
        self.true_label = None
        self.correct_prediction = None
        self.accuracy = None
        self.accuracy_summary = None
        self.merged = None
        self.x = None
        self.y_ = None
        self.score = None

        self.x = tf.placeholder("float", shape=[None, self.feature_size], name="input")
        self.y_ = tf.placeholder("float", shape=[None, self.label_size], name="label")
        with tf.name_scope(name="hidden-layer"):
            W = tf.Variable(tf.truncated_normal([self.feature_size, self.hidden_size], stddev=1.0), name="weights")
            b = tf.Variable(tf.constant(0.1, shape=[self.hidden_size]), name="bias")
            h = tf.sigmoid(tf.matmul(self.x, W, name="hidden") + b)
        with tf.name_scope(name="output-layer"):
            W = tf.Variable(tf.truncated_normal([self.hidden_size, self.label_size], stddev=1.0), name="weights")
            b = tf.Variable(tf.constant(0.1, shape=[self.label_size]), name="bias")
            y = tf.nn.softmax(tf.matmul(h, W) + b)
        self.score = y
        self.cross_entropy = -tf.reduce_sum(self.y_*tf.log(y))
        self.ce_summ = tf.scalar_summary("cross entropy", self.cross_entropy)
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)
        self.predict_label = tf.argmax(y, 1)
        self.true_label = tf.argmax(self.y_, 1)
        self.correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))
        self.accuracy_summary = tf.scalar_summary("accuracy", self.accuracy)

        # Merge all the summaries
        self.merged = tf.merge_all_summaries()
