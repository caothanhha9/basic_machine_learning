import bisect
import datetime
from data_collectors import DataCollector
from feature_calculators import FeatureCalculator
from utility.data_helpers import FileDAO
import numpy as np


class DataConverters(object):
    sess_id_ = 0
    timestamp_id_ = 1
    item_id_ = 2
    category_id_ = 3
    price_id_ = 3
    quantity_id_ = 4

    def __init__(self, file_path):
        self.file_path = file_path

    def convert_raw_data_to_features(self, _id_min=0L, _id_max=10000L, delimiter=',', current_item_id=-1, line_limit=2e6):
        def get_current_item_list(_data):
            _all_items = []
            for line_arr in _data:
                _all_items.append(line_arr[self.item_id_])
            reversed_data = _all_items[::-1]
            data_size = len(_data)
            _current_items = []
            _last_indexes = []
            for line_arr in _data:
                item = line_arr[self.item_id_]
                if not (item in _current_items):
                    _current_items.append(item)
                    last_index = data_size - 1 - reversed_data.index(item)
                    _last_indexes.append(last_index)
            return _current_items, _last_indexes
        print('start collecting click data')
        data_collector = DataCollector(file_path=self.file_path)
        id_list, data = data_collector.collect_data_reduce_by_id(_id_min=_id_min, _id_max=_id_max,
                                                                 delimiter=delimiter, line_limit=line_limit)
        print('start converting to features')
        feature_calculator = FeatureCalculator()
        feature_data = []
        current_item_list = []
        new_id_list = []
        for raw_data_id_, raw_data_ in enumerate(data):
            current_items, last_indexes = get_current_item_list(raw_data_)
            for curr_id_, curr_item_ in enumerate(current_items):
                new_id_list.append(id_list[raw_data_id_])
                current_item_id = last_indexes[curr_id_]
                # current_item_list.append(raw_data_[current_item_id][self.item_id_])
                current_item_list.append(curr_item_)
                feature_arr = feature_calculator.calculate_features(raw_data_, current_item_id=current_item_id)
                feature_data.append(feature_arr)
        return new_id_list, data, feature_data, current_item_list

    def convert_raw_data_to_labels(self, _id_list, _item_list, _id_min=0L, _id_max=10000L,
                                   delimiter=',', current_item_id=-1, limit=1e3, line_limit=2e6):
        print('start collecting buy data')
        file_dao = FileDAO(file_path=self.file_path)
        data = file_dao.get_all_lines_as_arrays(delimiter=delimiter)
        data.sort()
        session_list = [arr[0] for arr in data]

        print('start converting to labels')
        labels = []
        print(len(_id_list))
        for id_, session in enumerate(_id_list):
            print(id_)
            current_item = _item_list[id_]
            label = self.check_label(session=str(session), item=current_item, data=data, session_list=session_list)
            labels.append(label)
        del data
        return labels

    def check_label(self, session, item, data=None, session_list=None, delimiter=','):
        time_str = datetime.datetime.now().isoformat()
        print(time_str)
        label = 0
        if data is None:
            file_dao = FileDAO(file_path=self.file_path)
            data = file_dao.get_all_lines_as_arrays(delimiter=delimiter).sort()
            session_list = [arr[0] for arr in data]
        first_index = bisect.bisect_left(session_list, session)
        last_index = bisect.bisect_right(session_list, session)
        for line_arr in data[first_index:last_index]:
            if session == line_arr[0]:
                if item == line_arr[1]:
                    label = 1
            else:
                break
        time_str = datetime.datetime.now().isoformat()
        print(time_str)
        return label


def main():
    print('start')
    # file_path = '/media/hact/F8C6516EC6512DDE/Recommendation Engines/Email marketing/Linear and non-linear models for purchase prediction/yoochoose-dataFull/yoochoose-clicks.dat'
    file_path = '/media/cao/DATA/Study/Tech/Machine learning/Tech master/Purchase prediction/yoochoose-dataFull/yoochoose-clicks.dat'
    data_converter = DataConverters(file_path=file_path)
    id_list, data, feature_data, current_item_list = data_converter.convert_raw_data_to_features(
        _id_min=0L, _id_max=10000000000000L, line_limit=1e20)
    check_id = 1
    print(id_list[check_id])
    print(data[check_id])
    print(feature_data[check_id].get_feature_array())
    print(current_item_list[check_id])
    print('lengths')
    print(len(id_list))
    print(len(feature_data))
    print(len(current_item_list))
    file_path = '/media/cao/DATA/Study/Tech/Machine learning/Tech master/Purchase prediction/yoochoose-dataFull/yoochoose-buys.dat'
    del data_converter
    data_converter = DataConverters(file_path=file_path)
    labels = data_converter.convert_raw_data_to_labels(_id_list=id_list, _item_list=current_item_list,
                                                       _id_min=0L, _id_max=10000000000000L, limit=1e10, line_limit=1e20)
    print(labels[0])
    save_file_path = '../data/purchase_data'
    f = open(save_file_path, 'w')
    for id_, feature_ in enumerate(feature_data):
        label_ = labels[id_]
        feature_str = map(str, feature_.get_feature_array())
        save_line = '\t'.join(feature_str) + '\t' + str(label_)
        f.write(save_line)
        f.write('\n')
    f.close()


if __name__ == '__main__':
    main()
