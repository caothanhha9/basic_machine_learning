from sklearn import datasets


iris_data = datasets.load_iris()
iris_data_arr = iris_data.data  # or iris_data['data']
print('-' * 50)
print('iris data')
print('number of samples')
print(len(iris_data_arr))
print('feature size')
print(len(iris_data_arr[0]))
print('-' * 50)

digit_data = datasets.load_digits()
digit_data_arr = digit_data.data  # or digit_data['data']

print('digit data')
print('number of samples')
print(len(digit_data_arr))
print('feature size')
print(len(digit_data_arr[0]))
print('-' * 50)
