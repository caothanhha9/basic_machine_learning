import tensorflow as tf
from model import MLP
from data_loader import FileDAO, FeatureGen


def predict():
    file_dao = FileDAO("data/lottery_data")
    data = file_dao.get_data()
    feature_gen = FeatureGen(data)
    predict_data = feature_gen.get_predict_data(_feature_size=10)
    sess = tf.Session()
    mlp = MLP(feature_size=len(predict_data[0]), hidden_size=30, label_size=2)
    saver = tf.train.Saver(tf.all_variables())
    checkpoint_dir = 'models'
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        batch_xs = predict_data
        feed = {mlp.x: batch_xs}
        result = sess.run([mlp.score], feed_dict=feed)
        probs = [score[0] for score in result[0]]
        num_probs = list(zip(range(100), probs))
        num_probs.sort(key=lambda tup: tup[1], reverse=True)
        print(num_probs)


if __name__ == '__main__':
    predict()

