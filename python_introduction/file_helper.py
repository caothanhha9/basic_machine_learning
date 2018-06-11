# -*- coding: utf-8 -*-


def read_lines(file_path):
    # Mở file bằng lệnh open, option đặt là r - read
    f = open(file_path, 'r')
    # Đọc file bằng lệnh readlines
    lines = f.readlines()
    # Sau khi đọc xong hãy đóng file lại. Đây là thói quen tốt khi làm việc với io
    f.close()
    lines = [line.strip() for line in lines]
    return lines


def write_lines(lines, file_path):
    # Mở file bằng lệnh open, option đặt là w - write
    f = open(file_path, 'w')
    for line in lines:
        # Ghi lần lượt từng dòng
        f.write(line)
        # Hãy nhớ xuống dòng
        f.write('\n')
    # Sau khi viết xong hãy đóng file lại. Đây là thói quen tốt khi làm việc với io
    f.close()


def test_read_write():
    # Write lines to a file
    file_path = './test.txt'
    # Tạo ra một mảng các câu cần ghi vào file
    lines = ['welcome to machine learning', 'we will do well']
    # Dùng hàm viết file đã khai báo ở trên (hoặc ở dưới :D )
    write_lines(lines, file_path)
    # Read lines from a file
    loaded_lines = read_lines(file_path)
    for load_line in loaded_lines:
        print(load_line)

# Gọi hàm để kiểm tra việc viết và đọc file
test_read_write()
