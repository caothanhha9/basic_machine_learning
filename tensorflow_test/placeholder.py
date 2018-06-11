# -*- coding: utf-8 -*-
import tensorflow as tf
from sklearn import datasets


# Chuẩn bị dữ liệu để test: iris data
iris = datasets.load_iris()
data = iris.data
labels = iris.target

# Định nghĩa một placeholder với kích thước (shape) 10x4
# trong đó 1 là số lượng mẫu đưa vào 1 lượt tính, 4 là kích thước của 1 mẫu (feature_size)
x = tf.placeholder("float", shape=[1, 4], name="samples")
# y là một đại lượng cần tính thông qua input x
y = tf.multiply(x, 2, name="square")

# Tạo ra một session để tính toán
sess = tf.Session()

# Ví dụ ta muốn tính bình phương của 10 mẫu đầu tiên trong dữ liệu data
for i in range(10):
    x_input = [data[i]]
    print("input: ", x_input)
    bp = sess.run(y, feed_dict={x: x_input})
    print("square: ", bp)
