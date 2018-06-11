import bisect


class FileDAO(object):
    def __init__(self, address):
        self.address = address

    def get_data(self):
        f = open(self.address)
        lines = list(f.read().split("<eod>"))
        f.close()
        lines = [line.strip() for line in lines if len(line.strip()) > 0]
        data = []
        for line in lines:
            num_arr = line.split()
            num_arr = [num_.strip() for num_ in num_arr if len(num_.strip()) > 0]
            num_arr = map(int, num_arr)
            num_arr.sort()
            data.append(num_arr)
        return data


class FeatureGen(object):
    def __init__(self, data):
        self.data = data

    def get_feature_and_label(self, _number, _index=-1, _feature_size=10):
        feature_vec = []
        for num_arr in self.data[_index - _feature_size:_index]:
            left_id = bisect.bisect_left(num_arr, _number)
            if (left_id < 0) or (left_id >= len(num_arr)):
                feature_vec.append(0)
            elif num_arr[left_id] == _number:
                feature_vec.append(1)
            else:
                feature_vec.append(0)
        current_arr = self.data[_index]
        left_id = bisect.bisect_left(current_arr, _number)
        if (left_id < 0) or (left_id >= len(current_arr)):
            label = [0.0, 1.0]
        elif current_arr[left_id] == _number:
            label = [1.0, 0.0]
        else:
            label = [0.0, 1.0]
        return [feature_vec, label]

    def get_feature(self, _number, _index=-1, _feature_size=10):
        feature_vec = []
        end_index = _index + 1
        if end_index == 0:
            end_index = len(self.data)
        for num_arr in self.data[end_index - _feature_size:end_index]:
            left_id = bisect.bisect_left(num_arr, _number)
            if (left_id < 0) or (left_id >= len(num_arr)):
                feature_vec.append(0)
            elif num_arr[left_id] == _number:
                feature_vec.append(1)
            else:
                feature_vec.append(0)
        return feature_vec

    def get_train_data(self, _index_range=5, _feature_size=10, _offset=0):
        train_data = []
        train_labels = []
        for num_ in range(100):
            for i_ in range(_offset, _offset + _index_range):
                index_ = -1 - i_
                feature_, label_ = self.get_feature_and_label(num_, _index=index_, _feature_size=_feature_size)
                train_data.append(feature_)
                train_labels.append(label_)
        return [train_data, train_labels]

    def get_predict_data(self, _feature_size=10):
        predict_data = []
        for num_ in range(100):
            feature = self.get_feature(_number=num_, _feature_size=_feature_size)
            predict_data.append(feature)
        return predict_data


def main():
    print('start')
    file_dao = FileDAO("data/lottery_data")
    data = file_dao.get_data()
    feature_gen = FeatureGen(data)
    # print feature_gen.get_feature(_number=1, _index=-1, _feature_size=10)
    for num in data[-1]:
        print feature_gen.get_feature_and_label(_number=num, _index=-1, _feature_size=10)
    # train_data, train_labels = feature_gen.get_train_data()
    # for i in range(len(train_data)):
    #     print train_data[i]
    #     print train_labels[i]
    # print(len(train_data))

if __name__ == '__main__':
    main()
