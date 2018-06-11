import data_collectors
from utility import time_helpers


class Features(object):
    is_same_previous = None
    category_point = None
    previous_click_time = None
    is_most_click = None

    def __init__(self, feature_arr=None):
        if feature_arr is not None:
            self.is_same_previous = feature_arr[0]
            self.category_point = feature_arr[1]
            self.previous_click_time = feature_arr[2]
            self.is_most_click = feature_arr[3]

    def get_feature_array(self):
        return [self.is_same_previous, self.category_point, self.previous_click_time, self.is_most_click]


class FeatureCalculator(object):
    sess_id_ = 0
    timestamp_id_ = 1
    item_id_ = 2
    category_id_ = 3
    time_diff_thresh = 10 * 60  # 10 minutes
    most_click_thresh = float(1/3)
    click_count_thresh = 1

    def calculate_features(self, data, current_item_id=-1):
        data_size = len(data)
        if (data_size > 1) and (current_item_id != 0):
            item_list = []
            for line_arr in data:
                item = line_arr[self.item_id_]
                if not (item in item_list):
                    item_list.append(item)
            current_record = data[current_item_id]
            current_item = current_record[self.item_id_]
            current_category = current_record[self.category_id_]
            # Calculate is_same_previous
            previous_record = data[current_item_id - 1]
            previous_item = previous_record[self.item_id_]
            if current_item == previous_item:
                is_same_previous = 1.0
            else:
                is_same_previous = 0.0
            # Calculate previous_click_time
            current_time = current_record[self.timestamp_id_]
            previous_time = previous_record[self.timestamp_id_]
            previous_click_time = self.calculate_time_diff(previous_time, current_time)
            # Calculate category_point
            category_point = 0
            all_previous_data = data[:current_item_id]
            item_id_count_arr_ = [0 for _ in range(len(item_list))]
            for line_arr in all_previous_data:
                session = line_arr[self.sess_id_]
                timestamp = line_arr[self.timestamp_id_]
                item = line_arr[self.item_id_]
                category = line_arr[self.category_id_]
                if category == current_category:
                    category_point += 1
                link_id = item_list.index(item)
                item_id_count_arr_[link_id] += 1
            category_point = float(category_point + 1) / data_size
            # Calculate is_most_click

            def get_max_id(item_id_count_arr):
                max_count_id_list = []
                max_count = -1
                for count_num in item_id_count_arr:
                    if count_num > max_count:
                        max_count = count_num
                for count_id, count_num in enumerate(item_id_count_arr):
                    if count_num == max_count:
                        max_count_id_list.append(count_id)
                return max_count_id_list, max_count
            max_count_ids, max_count_ = get_max_id(item_id_count_arr_)
            current_item_id_ = item_list.index(current_item)
            if (current_item_id_ in max_count_ids) and (max_count_ > self.click_count_thresh):
                is_most_click = 1.0
                # Maybe consider change count / max_count or count / total_click
            else:
                is_most_click = 0.0
        else:
            is_same_previous = 0.0
            category_point = 0.0
            previous_click_time = 1.0
            is_most_click = 0.0
        current_feature = Features([is_same_previous, category_point, previous_click_time, is_most_click])
        return current_feature

    def calculate_time_diff(self, previous_time, current_time):
        time_diff_seconds = time_helpers.timestamp_diff_in_seconds(previous_time, current_time)
        time_diff = time_diff_seconds / self.time_diff_thresh
        if time_diff > 1.0:
            time_diff = 1.0
        return time_diff


def main():
    print('start...')
    feature_array = [0.0, 0.5, 0.2, 0.0]
    new_features = Features(feature_array)
    print new_features.get_feature_array()


if __name__ == '__main__':
    main()
