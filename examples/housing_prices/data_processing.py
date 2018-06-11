import pandas as pd


def load_hosing_price(file_path):
    df = pd.read_csv(file_path, header=0)
    return df


def test_load_data():
    df = load_hosing_price('train.csv')
    print('number of columns {}'.format(df.shape[1]))
    print('number of rows {}'.format(df.shape[0]))

if __name__ == '__main__':
    test_load_data()
