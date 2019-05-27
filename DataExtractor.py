from random import shuffle
import sys
import math
from Data.SNP import SNP
from Data.Sample import Sample
from utils import utils

c_parser = utils.c_parser


# Following two variables are prone to change
TOP_N_AHP_SNP = c_parser.getint('DATA_EXT', 'TOP_AHP')
TOP_N_SNP = c_parser.getint('DATA_EXT', 'TOP_PVAL')
PREFIX = c_parser.get('COMMON', 'prefix')

# Required to skip number of bytes while reading from plink binary file
# First two required to identify whether it is a plink file and 3rd one defines some kind of mode
# For more information go to: http://zzz.bwh.harvard.edu/plink/binary.shtml
SKIP_BYTES = 3

MASTER_THESIS_PATH = utils.MASTER_THESIS_PATH
AHP_path = MASTER_THESIS_PATH + 'Analiz/AHP/Output_SchGRU13/'
AHP_file = AHP_path + 'ahpscores.txt'

SNP_LIST = []
PATH = utils.PATH
file_name = 'merge_of_4'
binary_file = PATH + file_name + '.bed'
fam_file = PATH + file_name + '.fam'
map_file = PATH + file_name + '.bim'
p_file = PATH + file_name + '_p.qassoc.adjusted'


def get_snp_map():
    my_map = {}
    with open(map_file) as mf:
        for line in mf:
            splitted_line = line.split()
            snp = SNP()
            affy_name = splitted_line[1]
            snp.set_affy_name(affy_name)
            snp.set_bim_pos(len(my_map))
            my_map[affy_name] = snp
    return my_map

SNP_MAP = get_snp_map()


def get_top_n_snps():
    my_list = []
    with open(p_file) as pf:
        for i, line in enumerate(pf):
            if i == 0:  # it has header
                continue
            splitted_line = line.split()
            if len(splitted_line) != 10:
                print('Following line is not properly formatted:\n    {}'.format(line))
                continue
            snp = SNP_MAP[splitted_line[1]]
            my_list.append(snp)
            if len(my_list) == TOP_N_SNP:
                break
    return my_list


if TOP_N_SNP > 0:
    SNP_LIST.extend(get_top_n_snps())


def get_top_ahp_snps():
    my_list = []
    with open(AHP_file) as ahp_f:
        for line in ahp_f:
            splitted_line = line.split()
            affy_name = rsid_MAP[splitted_line[0]]
            snp = SNP_MAP[affy_name]
            my_list.append(snp)
            if len(my_list) == TOP_N_AHP_SNP:
                break
    return my_list


if TOP_N_AHP_SNP > 0:
    rsid_MAP = utils.get_rsid_map()
    SNP_LIST.extend(get_top_ahp_snps())

SNP_LIST = list(set(SNP_LIST))

with open(fam_file) as f:
    SAMPLE_LIST = [Sample(line.split()[-1]) for line in f]

NUM_OF_SAMPLES = len(SAMPLE_LIST)
if NUM_OF_SAMPLES == 0:
    print('Couldn\'t build sample list, exiting!')
    sys.exit(1)


def read_binary():
    '''
    1 byte is 8 bits which hold 4 genotypes for 4 samples for a single SNP
    Read binary file for given list of SNPs and finds genotype of the samples
    :param:
    :return:
    '''
    with open(binary_file, "rb") as bf:
        for snp in SNP_LIST:
            snp_block_size = int(math.ceil(NUM_OF_SAMPLES / 4))
            pos_to_seek = SKIP_BYTES + (snp_block_size * snp.get_bim_pos())
            end_pos = pos_to_seek + snp_block_size
            sample_no = 0
            for c_pos in range(pos_to_seek, end_pos):
                bf.seek(c_pos)
                read_bytes = bf.read(1)
                int_from_bytes = int.from_bytes(read_bytes, byteorder='little', signed=False)
                binary_str = '{0:08b}'.format(int_from_bytes)
                reversed_str = binary_str[::-1]
                for k in range(0, 8, 2):
                    sample = SAMPLE_LIST[sample_no]
                    sample_no += 1
                    sample.extend_genotype_data(reversed_str[k:k+2])
                    if sample_no == NUM_OF_SAMPLES:
                        break


read_binary()
case_list = [sample for sample in SAMPLE_LIST if sample.is_case()]
control_list = [sample for sample in SAMPLE_LIST if sample.is_control()]
train_size = int(math.floor(max(len(case_list), len(control_list)) * .9))


def write_to_file(fw):
    for phenotype_list in (case_list, control_list):
        shuffle(phenotype_list)
        if len(phenotype_list) > 0:
            sample = phenotype_list.pop()
            fw.write(sample.print_me_for_ANN() + '\n')


def write_model_files(file_type):
    with open('csvs/{}{}_{:d}.csv'.format(PREFIX, file_type, TOP_N_AHP_SNP+TOP_N_SNP), 'w') as fw:
        if file_type == 'train':
            for i in range(train_size):
                write_to_file(fw)
        elif file_type == 'test':
            while len(case_list) > 0 or len(control_list) > 0:
                write_to_file(fw)

[write_model_files(file_type) for file_type in ('train', 'test')]
