import glob
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from Utilities.Constants import SIZE_SET

cwd = os.getcwd()
for size in SIZE_SET[-4:-3]:
    for class_count_var in ['Limited', 'Complete']:
        for data_count_var in ['Partial', 'All']:
            print(class_count_var, data_count_var, size)
            os.chdir(cwd)
            os.chdir('..\\Results\\' + data_count_var + ' Data\\')
            result_files = list(glob.glob('PD *' + str(size) + '*' + class_count_var + '*.csv'))
            result_files = list(filter(lambda fil: 'Edited' not in fil, result_files))
            if len(result_files) < 3:
                print('Skipping')
                continue
            x = range(1, 26)
            oa_result_df = pd.DataFrame()
            for file in result_files:
                temp_df = pd.read_csv(file)
                hidden_layer = file.split(';')[0]
                originals = temp_df.columns
                for col in originals:
                    new_col_name = col.replace('_', ' ') + ' ' + hidden_layer
                    new_col_name = ' '.join(map(str.capitalize, new_col_name.split(' ')))
                    oa_result_df[new_col_name] = temp_df[col]

            sorted_columns = list(oa_result_df.columns)
            sorted_columns.sort()
            oa_result_df = oa_result_df.reindex(columns=sorted_columns)
            for value in ['Loss', 'Mean Absolute Error', 'Accuracy']:
                filtered_cols = list(filter(lambda z: value in z, oa_result_df.columns))
                filtered_df = oa_result_df[filtered_cols]

                plt.close('all')
                columns = list(filter(lambda z: value in z, oa_result_df.columns))
                if value != 'Accuracy':
                    y_limit = (0, np.round(filtered_df[columns].to_numpy().max() * 1.05, 3))
                else:
                    y_limit = (0, 1)
                title = value + '\nSize: ' + str(size) + ' ' + class_count_var + ' Classes' + ' ' + data_count_var + ' Data'
                filtered_df.plot(y=columns, title=title, ylim=y_limit, legend=True, xlabel='Epochs', ylabel=value)

                try:
                    os.chdir(cwd)
                    os.chdir('../Results\\')
                    plt.savefig('PD ' + title.replace(':', '').replace('\n', '\\') + '.png')
                except OSError:

                    os.mkdir(value)
                    plt.savefig('PD ' + title.replace(':', '').replace('\n', '\\') + '.png')
                os.chdir(cwd)
