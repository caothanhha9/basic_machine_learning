# -*- coding: utf-8 -*-
import tensorflow as tf

# Tạo ra một session
sess = tf.Session()

# Tạo ra một hằng số, kiểu string
hello = tf.constant('Hello, TensorFlow!')
sess.run(hello)

# Tạo ra một hằng số, kiểu int
a = tf.constant(10)
b = tf.constant(32)

# Tính toán trong một session (xem graph trong hình vẽ)
t = sess.run(a + b)
print(t)
