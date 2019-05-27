from utils import utils
import re

PATH = utils.PATH


def get_affy_rsid_map():
    conversion_file = PATH + 'old_analysis/conversion_map'
    my_map = {}

    with open(conversion_file) as cf:
        for line in cf:
            if '#' in line or 'Probe Set ID' in line:
                continue
            splitted_line = line.replace('"', '').split()
            rsid = splitted_line[1]
            if rsid == '---': continue
            affy_name = splitted_line[0]
            my_map[affy_name] = rsid

    # add missing rsid conversions
    conversion_file = utils.MASTER_THESIS_PATH + 'SNPmaster1-2-1.txt'
    with open(conversion_file) as cf:
        for i, line in enumerate(cf):
            if i == 0:
                continue
            splitted_line = line.split()
            affy_name = splitted_line[2]
            if affy_name not in my_map or my_map[affy_name] == '---':
                rsid = splitted_line[1]
                my_map[affy_name] = rsid
    return my_map


conversion_map = get_affy_rsid_map()

qassoc_file = PATH + 'merge_of_4.qassoc.adjusted'
formatted_qassoc_file = qassoc_file + '.formatted'

with open(qassoc_file) as qf, open(formatted_qassoc_file, 'w') as qfw:
    for i, line in enumerate(qf):
        if i == 0:
            qfw.write(line)
            continue
        splitted_line = re.compile('\s+').split(line.strip(' \n'))
        affy_name = splitted_line[1]
        if affy_name not in conversion_map: continue
        rsid = conversion_map[affy_name]
        if rsid == '---': continue
        formatted_line = '{:>4}{:>12}'.format(splitted_line[0], rsid)
        for k in range(2, len(splitted_line)):
            formatted_line += '{:>11}'.format(splitted_line[k])
        qfw.write(formatted_line + '\n')


