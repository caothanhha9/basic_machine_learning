# -*- coding: utf-8 -*-
"""
Có nhiều thư viện hỗ trợ sẵn các phép tính toán học
nhưng trong module này các phép tính được thực hiện
theo các công thức cơ bản với hai mục đích. Thứ nhất
ta có thể ôn lại các công thức toán học. Thứ hai ta
có thể luyện tập với cú pháp của python.
"""
import math


def cosine_sim(vector_a, vector_b):
    # Kích thước / số chiều của vector_a
    # (mặc định vector_a và vector_b có chiều dài bằng nhau)
    size_a = size_b = len(vector_a)
    # Tính tích vô hướng của hai vector
    dot_product = 0.0
    # i nhận các giá trị 0, 1, 2, ..., size_a - 1
    for i in range(size_a):
        # Tích vô hướng là tổng của tích các thành phần
        dot_product += vector_a[i] * vector_b[i]
    # Tính độ dài của vector a
    vector_a_mod = 0.0
    # i nhận các giá trị 0, 1, 2, ..., size_a - 1
    for i in range(size_a):
        vector_a_mod += vector_a[i] * vector_a[i]
    vector_a_mod = math.sqrt(vector_a_mod)
    # Tính độ dài của vector b
    vector_b_mod = 0.0
    # i nhận các giá trị 0, 1, 2, ..., size_b - 1
    for i in range(size_b):
        vector_b_mod += vector_b[i] * vector_b[i]
    vector_b_mod = math.sqrt(vector_b_mod)
    cos_sim = dot_product / (vector_a_mod * vector_b_mod)
    return cos_sim


def test_cosine_sim():
    a = [1, 2, 3]
    # a = [0, 1, 0]
    b = [4, 3, 2]
    # b = [2, 4, 6]
    # b = [0, 1, 1]
    print('a: {}, b: {}'.format(a, b))
    cos_similar = cosine_sim(a, b)
    print('cosine similarity: {}'.format(cos_similar))


def distance(vector_a, vector_b):
    # Kích thước / số chiều của vector_a
    # (mặc định vector_a và vector_b có chiều dài bằng nhau)
    size_a = size_b = len(vector_a)
    dist = 0.0
    # i nhận các giá trị 0, 1, 2, ..., size_a - 1
    for i in range(size_a):
        dist += math.pow(vector_a[i] - vector_b[i], 2)
    dist = math.sqrt(dist)
    return dist


def test_distance():
    x = [0, 1]
    y = [2, 3]
    z = [3, 2]
    dist_x_y = distance(x, y)
    dist_x_z = distance(x, z)
    dist_y_z = distance(y, z)
    print('x: {}, y: {}, z: {}'.format(x, y, z))
    print('Khoảng cách x, y')
    print(dist_x_y)
    print('Khoảng cách x, z')
    print(dist_x_z)
    print('Khoảng cách y, z')
    print(dist_y_z)


def dot_product(vector_a, vector_b):
    # Kích thước / số chiều của vector_a
    # (mặc định vector_a và vector_b có chiều dài bằng nhau)
    size_a = size_b = len(vector_a)
    # Tính tích vô hướng của hai vector
    dot_prd = 0.0
    # i nhận các giá trị 0, 1, 2, ..., size_a - 1
    for i in range(size_a):
        # Tích vô hướng là tổng của tích các thành phần
        dot_prd += vector_a[i] * vector_b[i]
    return dot_prd


def get_column(matrix_a, column_id):
    # Kích thước / số chiều của matrix_a
    a_row_size = len(matrix_a)
    a_col_size = len(matrix_a[0])
    column = []
    for i in range(a_row_size):
        column.append(matrix_a[i][column_id])
    return column


def matrix_multiply(matrix_a, matrix_b):
    # Kích thước / số chiều của matrix_a
    a_row_size = len(matrix_a)
    a_col_size = len(matrix_a[0])
    # Kích thước / số chiều của matrix_b
    b_row_size = len(matrix_b)
    b_col_size = len(matrix_b[0])
    # Tính tích của hai ma trận.
    matrix_product = []
    for i in range(a_row_size):
        matrix_p_row = []
        for j in range(b_col_size):
            product_el = dot_product(matrix_a[i], get_column(matrix_b, j))
            matrix_p_row.append(product_el)
        matrix_product.append(matrix_p_row)
    return matrix_product


def test_matrix_multiply():
    a = [[1.2, 0.8], [1.4, 0.9], [1.5, 1.0], [1.2, 0.8], [1.4, 0.5]]
    b = [[1.5, 1.2, 1.0, 0.8], [1.7, 0.6, 1.1, 0.4]]
    matrix_p = matrix_multiply(a, b)
    print('a: {}'.format(a))
    print('b: {}'.format(b))
    print('a x b')
    print(matrix_p)

if __name__ == '__main__':
    # test_cosine_sim()
    # test_distance()
    test_matrix_multiply()
