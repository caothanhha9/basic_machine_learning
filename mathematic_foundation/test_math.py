from optparse import OptionParser
from algebra import test_distance, test_cosine_sim, test_matrix_multiply


parser = OptionParser()

parser.add_option('-s', '--subfield', dest='sub_field',
                  help='Field to test. al = algebra, an = analysis, st = statistic')
parser.add_option('-f', '--function', dest='function',
                  help='Function to test. dist = distance, cos = cosine, matmul = matrix multiply')


(options, args) = parser.parse_args()

try:
    if options.sub_field == 'al':
        if options.function == 'dist':
            test_distance()
        elif options.function == 'cos':
            test_cosine_sim()
        elif options.function == 'matmul':
            test_matrix_multiply()
except:
    parser.print_help()
    pass

