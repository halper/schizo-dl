
ENCODING_MULTIPLIER = 2 # how many data points each column contains
TOP_n = 60000 # number of top columns that are going to be used for data extraction

# Read from importances file to get column numbers
importances_file = 'results/importances'

column_numbers = []
with open(importances_file) as f:
    for i, line in enumerate(f):
        line = line.split('\t')[0]
        col_num = line.replace('SNP_', '')
        col_num = int(col_num)
        # Subtracting 1 from the col number because I want it to be consistent with array indexing
        column_numbers.append(col_num - 1)
        if i + 1 == TOP_n:
            break

# Sort the column numbers asc
column_numbers = sorted(column_numbers)
print(column_numbers)

# Extract the data from plink files accordingly with the related columns
test_file_to_be_read = 'csvs/pval_enc_test_86250.csv'
test_file_to_be_written = 'csvs/RF_prioritized_enc_test_{}.csv'.format(TOP_n)
train_file_to_be_read = 'csvs/pval_enc_train_86250.csv'
train_file_to_be_written = 'csvs/RF_prioritized_enc_train_{}.csv'.format(TOP_n)

files = [(test_file_to_be_read, test_file_to_be_written), (train_file_to_be_read, train_file_to_be_written)]

for fr, fw in files:
    with open(fr) as reading_file, open(fw, 'w') as writing_file:
        for line in reading_file:
            splitted_line = line.strip('\n').split(',')
            new_line = splitted_line.pop(0) + ',' # first is label case/control
            for i, cn in enumerate(column_numbers):
                cn = cn * ENCODING_MULTIPLIER
                new_line += '{},{}'.format(splitted_line[cn], splitted_line[cn+1])
                if i + 1 != len(column_numbers):
                    new_line += ','
            writing_file.write(new_line + '\n')
