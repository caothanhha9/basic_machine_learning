# -*- coding: utf-8 -*-
import os
import time
from utils import data_helpers
import numpy as np
import tensorflow as tf
from rnn_model import PTBModel
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
    print(data)
    size_limit = 30
    find_stop_word = False
    state = m.initial_state.eval()
    current_step = 0
    gen_words = []
    input_data = data
    while (current_step < size_limit) and (not find_stop_word):
        state, probs, logits, _ = session.run([m.final_state, m.probabilities, m.logits, eval_op],
                                              {m.input_data: input_data,
                                              m.initial_state: state})
        chosen_word = np.argmax(probs, 1)
        print("Probabilities shape: %s, Logits shape: %s" %
              (probs.shape, logits.shape) )
        print(chosen_word)
        next_word = 'unk'
        next_word_id = 0
        if vocabulary is not None:
            next_word_id = chosen_word[-1]
            for word_, word_id_ in vocabulary.iteritems():
                if word_id_ == next_word_id:
                    print(word_)
                    next_word = word_
        # next_word_arr = next_word.split('<eos>')
        # if len(next_word_arr) > 1:
        #     next_word = next_word_arr[0]
        #     find_stop_word = True
        if next_word == '<eos>':
            find_stop_word = True
        gen_words.append(next_word)
        input_data = np.array(input_data)[:, 1:]
        input_data = np.concatenate((input_data, np.array([[next_word_id]])), axis=1)
        current_step += 1
    return gen_words


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
    data_path = FilePath.data_path
    model_option = Config.model_option
    if not data_path:
        raise ValueError("Must set --data_path to PTB data directory")

    out_dir = 'models'
    checkpoint_dir = os.path.join(out_dir, "checkpoints")

    raw_data = data_helpers.raw_data(data_path)
    train_data, valid_data, test_data, vocabulary = raw_data

    config = get_config(model_option)
    eval_config = get_config(model_option)
    eval_config.batch_size = 1
    # eval_config.num_steps = 1

    # sentence = 'Bạn khỏe không? <eos>'
    # sentence = 'Bạn khỏe'
    # sentence = 'Anh khỏe không? <eos>'
    # sentence = 'Chị khỏe không? <eos>'
    sentence = 'Chị ăn cơm chưa? <eos>'
    predict_data = data_helpers.predict_data(sentence, vocabulary, config.num_steps)
    eval_config.num_steps = len(predict_data[0])

    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)

        with tf.variable_scope("model", reuse=None, initializer=initializer):
            m = PTBModel(is_training=True, config=config)
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            mvalid = PTBModel(is_training=False, config=config)
            mtest = PTBModel(is_training=False, config=eval_config)
            mpredict = PTBModel(is_training=False, config=eval_config)

        # tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=1)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            model_checkpoint_path_arr = ckpt.model_checkpoint_path.split("/")
            abs_model_checkpoint_path = checkpoint_dir + '/' + model_checkpoint_path_arr[-1]
            saver.restore(session, abs_model_checkpoint_path)

        gen_words = run_epoch(session, mpredict, predict_data, tf.no_op(), vocabulary=vocabulary)
        print('=' * 50)
        print('=' * 50)
        print(sentence)
        print(' '.join(gen_words))


if __name__ == "__main__":
    main()
