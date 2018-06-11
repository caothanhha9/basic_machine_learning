# -*- coding: utf-8 -*-
import tensorflow as tf
from sklearn import datasets


# Định nghĩa một biến với kích thước (shape) 4x5
x = tf.Variable(tf.zeros([1, 4]))
inc_op = x.assign(tf.add(x, [1.0, 1.0, 1.0, 1.0]))

# Operator: Gán cho các biến giá trị khởi tạo
init = tf.initialize_all_variables()

# Tạo ra một session để tính toán
sess = tf.Session()
sess.run(init)  # Execute init = Gán cho các biến giá trị khởi tạo


# Tăng giá trị của biến sau mỗi vòng lặp
for i in range(10):
    curr_val, _ = sess.run([x, inc_op])
    print(curr_val)
