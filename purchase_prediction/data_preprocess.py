from utility import data_helpers


class DataProcessor(object):
    def __init__(self, file_path):
        self.name = 'Data Processor'
        self.file_path = file_path
        self.out_file_path = file_path + '.processed'

    def remove_default(self):
        default_line = "0.0\t0.0\t1.0\t0.0\t1"
        f = open(self.file_path, "r")
        lines = f.readlines()
        f.close()
        lines = [s.strip() for s in lines]
        fw = open(self.out_file_path, "w")
        for line in lines:
            if line != default_line:
                fw.write(line)
                fw.write("\n")
        fw.close()


def main():
    print('start')
    file_path = "../data/purchase_data_1000"
    data_processor = DataProcessor(file_path)
    data_processor.remove_default()


if __name__ == '__main__':
    main()
