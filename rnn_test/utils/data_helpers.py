# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# pylint: disable=unused-import,g-bad-import-order

"""Utilities for parsing PTB text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys
import time

import tensorflow.python.platform

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.python.platform import gfile


def _read_words(filename):
    with gfile.GFile(filename, "r") as f:
        return f.read().replace("\n", " <eos> ").split()


def _build_vocab(filename):
    data = _read_words(filename)
    counter = collections.Counter(data)
    # counter.update({'unk': -1})
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    count_pairs = [('unk', -1)] + count_pairs

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    return word_to_id


def _file_to_word_ids(filename, word_to_id):
    data = _read_words(filename)
    convert_word_to_id = [word_to_id[word] if word in word_to_id else word_to_id['unk'] for word in data]
    return convert_word_to_id


def raw_data(data_path=None):
    """Load PTB raw data from data directory "data_path".

    Reads PTB text files, converts strings to integer ids,
    and performs mini-batching of the inputs.

    The PTB dataset comes from Tomas Mikolov's webpage:

    http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

    Args:
      data_path: string path to the directory where simple-examples.tgz has
        been extracted.

    Returns:
      tuple (train_data, valid_data, test_data, vocabulary)
      where each of the data objects can be passed to PTBIterator.
    """

    train_path = os.path.join(data_path, "train.txt")
    valid_path = os.path.join(data_path, "valid.txt")
    test_path = os.path.join(data_path, "test.txt")

    word_to_id = _build_vocab(train_path)
    train_data = _file_to_word_ids(train_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)
    # vocabulary = len(word_to_id)
    vocabulary = word_to_id
    return train_data, valid_data, test_data, vocabulary


def predict_data(sentence, word_to_id, num_steps):
    data = sentence.split()
    convert_word_to_id = [word_to_id[word] if word in word_to_id else word_to_id['unk'] for word in data]
    pad_data = np.zeros([num_steps], dtype=np.int32)
    crop_data = convert_word_to_id[-num_steps:]
    len_crop = len(crop_data)
    if len_crop < num_steps:
        step_data = [np.concatenate((pad_data[:num_steps - len_crop], crop_data), axis=0)]
    else:
        step_data = [crop_data]
    return [convert_word_to_id]
    # return step_data


def iterator(raw_data_, batch_size, num_steps):
    """Iterate on the raw PTB data.

    This generates batch_size pointers into the raw PTB data, and allows
    minibatch iteration along these pointers.

    Args:
      raw_data_: one of the raw data outputs from raw_data.
      batch_size: int, the batch size.
      num_steps: int, the number of unrolls.

    Yields:
      Pairs of the batched data, each a matrix of shape [batch_size, num_steps].
      The second element of the tuple is the same data time-shifted to the
      right by one.

    Raises:
      ValueError: if batch_size or num_steps are too high.
    """
    raw_data_ = np.array(raw_data_, dtype=np.int32)

    data_len = len(raw_data_)
    # if batch_size > data_len:
    #     if data_len % 2 == 0:
    #         batch_size = data_len / 2
    #     else:
    #         batch_size = data_len // 3
    batch_len = data_len // batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
        data[i] = raw_data_[batch_len * i:batch_len * (i + 1)]

    epoch_size = (batch_len - 1) // num_steps
    print(batch_len)
    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        x = data[:, i*num_steps:(i+1)*num_steps]
        y = data[:, i*num_steps+1:(i+1)*num_steps+1]
        yield (x, y)


