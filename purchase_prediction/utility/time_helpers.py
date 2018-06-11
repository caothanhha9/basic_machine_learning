import datetime
import time


epoch = datetime.datetime.utcfromtimestamp(0)


def unix_time_millis(dt):
    return (dt - epoch).total_seconds() * 1000.0


def timestamp_to_seconds(time_stamp_str, time_format='%Y-%m-%dT%H:%M:%S.%fZ'):
    time_struct = time.strptime(time_stamp_str, time_format)
    return time.mktime(time_struct)


def timestamp_diff_in_seconds(time_stamp_str0, time_stamp_str1):
    time0 = timestamp_to_seconds(time_stamp_str0)
    time1 = timestamp_to_seconds(time_stamp_str1)
    time_diff = time1 - time0
    return time_diff


def second_to_minute(seconds):
    return seconds / 60


def main():
    print('start...')
    dt = '2014-04-07T10:51:09.277Z'
    print timestamp_to_seconds(dt)
    dt1 = '2014-04-07T10:51:10.277Z'
    time_diff = timestamp_diff_in_seconds(dt, dt1)
    print(time_diff)
    print second_to_minute(time_diff)


if __name__ == '__main__':
    main()
