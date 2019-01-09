from random import shuffle
import sys
import subprocess
import time
import math


class Sample:
    CONTROL = '1'
    CASE = '2'

    def __init__(self, phenotype):
        self.phenotype = str(phenotype)
        self.genotype_data = ''

    def is_control(self):
        return self.phenotype == self.CONTROL

    def is_case(self):
        return self.phenotype == self.CASE

    def extend_genotype_data(self, genotype):
        self.genotype_data += ',{:n}'.format(genotype)

    def print_me_for_ANN(self):
        phenotype = '0' if self.is_control() else '1' if self.is_case() else ''
        return '{}{}'.format(phenotype, self.genotype_data)


class SNP:
    def __init__(self):
        self.bim_pos = 0
        self.affy_name = ''
        self.rsid = ''

    def set_affy_name(self, affy_name):
        self.affy_name = affy_name

    def get_affy_name(self):
        return self.affy_name

    def set_rsid(self, rsid):
        self.rsid = rsid

    def set_bim_pos(self, bim_pos):
        self.bim_pos = int(bim_pos)

    def get_bim_pos(self):
        return self.bim_pos


SKIP_BYTES = 3
path = "/Users/alper/Documents/tez/analysis/" # "/Volumes/Untitled/Tez Data/data/"
file_name = 'merge_of_4'
binary_file = path + file_name + '.bed'
fam_file = path + file_name + '.fam'
map_file = path + file_name + '.bim'
p_file = path + file_name + '_p.assoc.adjusted'

MASTER_THESIS_PATH = '/Users/alper/Dropbox/Google Drive/AydınSon Lab/Master Tezi/'

TOP_N_SNP = 500
SNP_LIST = []


def get_rsid_map():
    conversion_file = '/Volumes/Untitled/GenomeWideSNP_6.na35.annot.csv'
    my_map = {}
    with open(conversion_file) as cf:
        for line in cf:
            if '#' in line or 'Probe Set ID' in line:
                continue
            splitted_line = line.replace('"', '').split(',')
            rsid = splitted_line[1]
            affy_name = splitted_line[0]
            my_map[rsid] = affy_name

    #add missing rsid conversions
    conversion_file = MASTER_THESIS_PATH + 'SNPmaster1-2-1.txt'
    with open(conversion_file) as cf:
        for i, line in enumerate(cf):
            if i == 0:
                continue
            splitted_line = line.split()
            rsid = splitted_line[1]
            if rsid not in my_map:
                affy_name = splitted_line[2]
                my_map[rsid] = affy_name
    return my_map

rsid_MAP = get_rsid_map()
AHP_path = MASTER_THESIS_PATH + 'Analiz/AHP/Output_SchGRU13/'
AHP_file = AHP_path + 'ahpscores.txt'

TOP_N_AHP_SNP = 500


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
    SNP_LIST.extend(get_top_ahp_snps())

with open(fam_file) as f:
    SAMPLE_LIST = [Sample(line.split()[-1]) for line in f]

NUM_OF_SAMPLES = len(SAMPLE_LIST)
if NUM_OF_SAMPLES == 0:
    print('Couldn\'t build sample list, exiting!')
    sys.exit(1)


def read_binary(binary_file):
    '''
    1 byte is 8 bits which hold 4 genotypes for 4 samples for a single SNP
    :param binary_file:
    :return:
    '''
    with open(binary_file, "rb") as bf:
        '''
        bf.seek(0, 2)  # Seek the end
        num_bytes = bf.tell()  # Get the file size
        '''
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
                    genotype = int(reversed_str[k:k+2], 2)
                    sample.extend_genotype_data(genotype)
                    if sample_no == NUM_OF_SAMPLES:
                        break


read_binary(binary_file)
case_list = [sample for sample in SAMPLE_LIST if sample.is_case()]
control_list = [sample for sample in SAMPLE_LIST if sample.is_control()]
shuffle(case_list)
shuffle(control_list)
train_size = int(math.floor(max(len(case_list), len(control_list)) * .7))

with open('train_{:d}.csv'.format(TOP_N_AHP_SNP+TOP_N_SNP), 'w') as fw:
    for i in range(train_size):
        if len(case_list) > 0:
            sample = case_list.pop()
            fw.write(sample.print_me_for_ANN() + '\n')
        if len(control_list) > 0:
            sample = control_list.pop()
            fw.write(sample.print_me_for_ANN() + '\n')

with open('test_{:d}.csv'.format(TOP_N_AHP_SNP+TOP_N_SNP), 'w') as fw:
    while len(case_list) > 0 or len(control_list) > 0:
        if len(case_list) > 0:
            sample = case_list.pop()
            fw.write(sample.print_me_for_ANN() + '\n')
        if len(control_list) > 0:
            sample = control_list.pop()
            fw.write(sample.print_me_for_ANN() + '\n')
