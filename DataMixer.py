import random
from Data.Sample import Sample

file_mixed = 'csvs/pval_enc_mixed_test_86250.csv'

file_to_be_mixed = 'csvs/pval_enc_test_86250.csv'

with open(file_to_be_mixed) as f, open(file_mixed, 'w') as fw:
    for line in f:
        phenotype = line.split(',')[0]
        if phenotype == '1':
            rand_pheno = str(random.randint(0, 1))
            fw.write(rand_pheno + line[1:])
        else:
            fw.write(line)