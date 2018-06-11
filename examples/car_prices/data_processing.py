import numpy as np
from matplotlib import pyplot as plt
import random
import pandas as pd


def gen():
    x = np.asarray(range(10, 120000, 500))
    y = []

    with open('car_price2.txt', 'w') as fw:
        for xi in x:
            delta_y = random.randint(-5000, 1000)
            yi = (xi - 140000) ** 2 / 1399800 + 3000 + delta_y
            y.append(yi)
            fw.write('{},{}\n'.format(xi, yi))
    print(x.shape)
    plt.scatter(x, y)
    plt.show()


def load_car_price(file_path):
    column_names = ['mileage', 'price']
    df = pd.read_csv(file_path, header=0, names=column_names)
    x = df['mileage'].as_matrix()
    y = df['price'].as_matrix()
    return x, y


def test_load_data():
    file_path = 'car_price.csv'
    x, y = load_car_price(file_path)
    print(x)
    print(y)

if __name__ == '__main__':
    gen()
    #test_load_data()

