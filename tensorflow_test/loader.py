# -*- coding: utf-8 -*-
import tensorflow as tf


# Định nghĩa một biến với kích thước (shape) 4x5
W = tf.Variable(tf.zeros([1, 4]), name="W")
inc_op = W.assign(tf.add(W, [1.0, 1.0, 1.0, 1.0]))

# Operator: Gán cho các biến giá trị khởi tạo
init = tf.initialize_all_variables()

# Tạo ra một session để tính toán
sess = tf.Session()

# Tạo ra saver object để lưu model vào file
saver = tf.train.Saver(tf.all_variables(), max_to_keep=1)
# tf.all_variables() các object được lưu là tất cả các biến trong tf,
#  max_to_keep số file được lưu tối đa cho 1 model

# Đường dẫn để lưu model, trong đó .../models là thư mục và model là prefix của tên file
checkpoint_dir = 'models'

sess.run(init)  # Execute init = Gán cho các biến giá trị khởi tạo

ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
    check_W = sess.run(W)
    print(check_W)
