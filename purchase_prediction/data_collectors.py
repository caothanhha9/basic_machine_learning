from utility.data_helpers import FileDAO
import os


class DataCollector(object):
    def __init__(self, file_path):
        self.name = 'DataCollector'
        self.file_path = file_path

    def collect_data_by_id(self, _id, _array_id=0, delimiter='\t',
                           _batch_num=10000, limit=1e6, not_found_next_limit=1e6,
                           not_found_limit=1e6):
        data = []
        found_count = 0
        batch_count = 0
        not_found_next_count = 0
        not_found_count = 0
        file_dao = FileDAO(self.file_path)
        offset = 0
        eof = False
        while (found_count < limit) and (not eof) and (not_found_next_count < not_found_next_limit)\
                and (not_found_count < not_found_limit):
            lines, eof = file_dao.get_range_lines(offset, _batch_num)
            for line in lines:
                line_arr = line.split(delimiter)
                # if _id in line:
                #     data.append(line)
                if line_arr[_array_id] == _id:
                    data.append(line_arr)
                    found_count += 1
                    not_found_count = 0  # note
                else:
                    not_found_count += 1
                    print(not_found_count)
                    if found_count > 0:
                        not_found_next_count += 1
            offset += _batch_num
            batch_count += 1
        return data

    def collect_id_list(self, _array_id=0, _id_limit=1e3, _batch_num=10000, delimiter='\t',
                        limit=1e3, line_limit=2e6):
        data = []
        file_dao = FileDAO(self.file_path)
        batch_count = 0
        offset = 0
        eof = False
        found_count = 0
        line_count = 0
        while (found_count < limit) and (line_count < line_limit) and (not eof):
            lines, eof = file_dao.get_range_lines(offset, _batch_num)
            for line in lines:
                line_arr = line.split(delimiter)
                id_ = long(line_arr[_array_id])
                # if _id in line:
                #     data.append(line)
                if (not (id_ in data)) and (id_ < _id_limit):
                    data.append(id_)
                    found_count += 1
                line_count += 1
            offset += _batch_num
            batch_count += 1
        return data

    def collect_data_reduce_by_id(self, _array_id=0, _id_min=0L, _id_max=10000L, _batch_num=10000, delimiter='\t',
                                  limit=1e3, line_limit=2e6):
        data = []
        id_list = []
        file_dao = FileDAO(self.file_path)
        batch_count = 0
        offset = 0
        eof = False
        found_count = 0
        line_count = 0
        while (found_count < limit) and (line_count < line_limit) and (not eof):
            lines, eof = file_dao.get_range_lines(offset, _batch_num)
            print(offset)
            for line in lines:
                line_arr = line.split(delimiter)
                id_ = long(line_arr[_array_id])
                # if _id in line:
                #     data.append(line)
                if not (id_ in id_list):
                    if _id_min <= id_ <= _id_max:
                        id_list.append(id_)
                        data.append([line_arr])
                        found_count += 1
                else:
                    link_id = id_list.index(id_)
                    data[link_id].append(line_arr)
                line_count += 1
            offset += _batch_num
            batch_count += 1
        return id_list, data


def main():
    print('start...')
    # file_path = '../data/customer_saving_salary'
    # file_path = '/media/cao/DATA/Study/Tech/Machine learning/Tech master/Purchase prediction/yoochoose-dataFull/yoochoose-clicks.dat'
    # file_path = '/media/hact/F8C6516EC6512DDE/Recommendation Engines/Email marketing/Linear and non-linear models for purchase prediction/yoochoose-dataFull/yoochoose-clicks.dat'
    # print os.path.isfile(file_path)
    # data_collector = DataCollector(file_path=file_path)
    # data = data_collector.collect_data_by_id(_id='33', delimiter=',', limit=1e3,
    #                                          not_found_next_limit=1e3, not_found_limit=1e5)
    # print(data)
    # print(len(data))

    # file_path = '/media/hact/F8C6516EC6512DDE/Recommendation Engines/Email marketing/Linear and non-linear models for purchase prediction/yoochoose-dataFull/yoochoose-buys.dat'
    # data_collector = DataCollector(file_path=file_path)
    # id_list = data_collector.collect_id_list(delimiter=',', _id_limit=1e4)
    # print(id_list)
    # print(len(id_list))

    # file_path = '/media/hact/F8C6516EC6512DDE/Recommendation Engines/Email marketing/Linear and non-linear models for purchase prediction/yoochoose-dataFull/yoochoose-buys.dat'
    # file_path = '/media/hact/F8C6516EC6512DDE/Recommendation Engines/Email marketing/Linear and non-linear models for purchase prediction/yoochoose-dataFull/yoochoose-clicks.dat'
    file_path = '/media/cao/DATA/Study/Tech/Machine learning/Tech master/Purchase prediction/yoochoose-dataFull/yoochoose-clicks.dat'
    data_collector = DataCollector(file_path=file_path)
    id_list, data = data_collector.collect_data_reduce_by_id(_id_min=0L, _id_max=10000000000000L, delimiter=',',
                                                             limit=1e10, line_limit=1e20)
    print(id_list)
    print(data)
    print(id_list[0])
    print(data[0])


if __name__ == '__main__':
    main()

