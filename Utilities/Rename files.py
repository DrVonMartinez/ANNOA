import glob
import os
import pandas

os.chdir('../Results' + '\\')


def replace(data):
    data_set = data.split(';')
    sequence = ''
    for i in range(len(data_set)):
        if i in [0, 1, 4, 5]:
            sequence += data_set[i]
        if i in [0, 1]:
            sequence += ','
    return sequence


cwd = os.getcwd()
for class_count_var in ['Limited Classes', 'Full Classes']:
    for data_count_var in ['Partial Data', 'All Data']:
        os.chdir(cwd)
        os.chdir(data_count_var + '\\' + class_count_var + '\\')
        test_list = list(glob.glob('PD*,*.csv'))
        test_list.sort()
        for pd_name in test_list:
            pd_df = pandas.read_csv(pd_name)
            new_name = replace(pd_name).replace('PD', 'Dim 2,')[:-1]
            pd_df.to_csv(new_name, index=False)
        os.chdir(cwd)
