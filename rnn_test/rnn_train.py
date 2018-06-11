import os
import time
import numpy as np
import tensorflow as tf
from rnn_model import PTBModel
from utils import data_helpers
from utils.config import SmallConfig, MediumConfig, LargeConfig, FilePath, Config


def run_epoch(session, m, data, eval_op, verbose=False, vocabulary=None):
    """
    :param session for computation
    :param m model object
    :param data input data
    :param eval_op
    :param verbose
    :param vocabulary
    Runs the model on the given data."""
    epoch_size = ((len(data) // m.batch_size) - 1) // m.num_steps
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = m.initial_state.eval()
    for step, (x, y) in enumerate(data_helpers.iterator(data, m.batch_size,
                                                          m.num_steps)):
        print('-x-y-' * 50)
        print(x)
        print(y)
        cost, state, probs, logits, _ = session.run([m.cost, m.final_state, m.probabilities, m.logits, eval_op],
                                                    {m.input_data: x,
                                                     m.targets: y,
                                                     m.initial_state: state})
        costs += cost
        iters += m.num_steps
        if epoch_size // 10 > 0:  # hact added
            if verbose and step % (epoch_size // 10) == 10:
                print("%.3f perplexity: %.3f speed: %.0f wps" %
                      (step * 1.0 / epoch_size, np.exp(costs / iters),
                       iters * m.batch_size / (time.time() - start_time)))
                chosen_word = np.argmax(probs, 1)
                print("Probabilities shape: %s, Logits shape: %s" %
                      (probs.shape, logits.shape) )
                print(chosen_word)
                if vocabulary is not None:
                    next_word_id = chosen_word[-1]
                    for word_, word_id_ in vocabulary.iteritems():
                        if word_id_ == next_word_id:
                            print(word_)

                print("Batch size: %s, Num steps: %s" % (m.batch_size, m.num_steps))

    return np.exp(costs / iters)


def get_config(model_option):
    if model_option == "small":
        return SmallConfig()
    elif model_option == "medium":
        return MediumConfig()
    elif model_option == "large":
        return LargeConfig()
    else:
        raise ValueError("Invalid model: %s", model_option)


def main():
    # --data_path=/tmp/simple-examples/data/ --model small
    # data_path = '/home/hact/workspace/python/machinelearning/rnn_test/data/'
    data_path = FilePath.data_path
    # link to download zip file simple-examples: http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
    # model_option = 'small'
    model_option = Config.model_option
    if not data_path:
        raise ValueError("Must set --data_path to PTB data directory")

    out_dir = 'models'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    checkpoint_dir = os.path.join(out_dir, "checkpoints")
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    raw_data = data_helpers.raw_data(data_path)
    train_data, valid_data, test_data, vocabulary = raw_data

    config = get_config(model_option)
    eval_config = get_config(model_option)
    eval_config.batch_size = 1
    eval_config.num_steps = 1

    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            m = PTBModel(is_training=True, config=config)
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            mvalid = PTBModel(is_training=False, config=config)
            mtest = PTBModel(is_training=False, config=eval_config)

        saver = tf.train.Saver(tf.all_variables(), max_to_keep=1)
        tf.initialize_all_variables().run()

        for i in range(config.max_max_epoch):
            lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
            m.assign_lr(session, config.learning_rate * lr_decay)

            print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
            train_perplexity = run_epoch(session, m, train_data, m.train_op,
                                         verbose=True, vocabulary=vocabulary)
            print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
            # hact commented
            # valid_perplexity = run_epoch(session, mvalid, valid_data, tf.no_op(), vocabulary=vocabulary)
            # print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

            path = saver.save(session, checkpoint_prefix, global_step=i)
            print('save to: ' + str(path))

            # test_perplexity = run_epoch(session, mtest, test_data, tf.no_op(), vocabulary=vocabulary)
            # print("Test Perplexity: %.3f" % test_perplexity)


if __name__ == "__main__":
    main()
