import glob
import os
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

from Utilities.Constants import SIZE_SET, METRIC_SET, SHOW
from Utilities.Tensor_Constants import NUM_HIDDEN_LAYERS


def sort(val):
    return int(val.split(' ')[-2])


cwd = os.getcwd()
for class_count_var in ['Full Classes']:
    if class_count_var == 'Full Classes':
        accuracy = [1/49] * 50
    else:
        accuracy = [1/7] * 50
    for data_count_var in ['All Data', 'Partial Data']:
        os.chdir(cwd)
        os.chdir('..\\Results\\' + data_count_var + '\\' + class_count_var + '\\')
        print(os.getcwd())
        for num_hidden_layer in NUM_HIDDEN_LAYERS:
            pandas_obj_list = list(glob.glob('PD*Layer ' + str(num_hidden_layer) + ';*'))
            pandas_converted_list = list(glob.glob('PD*Layer ' + str(num_hidden_layer) + ',*'))
            reduced_list = list(glob.glob('REDUCED*Layer ' + str(num_hidden_layer) + '*'))
            test_list = list(glob.glob('TEST*Layer ' + str(num_hidden_layer) + '*'))
            pandas_obj_list.sort()
            pandas_converted_list.sort()
            reduced_list.sort()
            test_list.sort()
            pandas_obj_df_list = []
            pandas_converted_df_list = []
            reduced_df_list = []
            test_df_list = []
            for pandas_obj, size in zip(pandas_obj_list, SIZE_SET):
                pandas_obj_df = pd.read_csv(pandas_obj)
                pandas_obj_columns = [column + ' pd_obj ' + str(size) + ' ' for column in pandas_obj_df.columns]
                pandas_obj_df_list.append(pandas_obj_df.set_axis(pandas_obj_columns, axis=1))
            for pandas_converted, size in zip(pandas_converted_list, SIZE_SET):
                pandas_converted_df = pd.read_csv(pandas_converted)
                pandas_converted_columns = [column + ' pd_converted ' + str(size) + ' ' for column in pandas_converted_df.columns]
                pandas_converted_df_list.append(pandas_converted_df.set_axis(pandas_converted_columns, axis=1))
            for reduced, size in zip(reduced_list, SIZE_SET):
                reduced_df = pd.read_csv(reduced)
                reduced_columns = [column + ' reduced ' + str(size) + ' ' for column in reduced_df.columns]
                reduced_df_list.append(reduced_df.set_axis(reduced_columns, axis=1))
            for test, size in zip(test_list, SIZE_SET):
                test_df = pd.read_csv(test)
                test_columns = [column + ' test ' + str(size) + ' ' for column in test_df.columns]
                test_df_list.append(test_df.set_axis(test_columns, axis=1))
            temp_list = pandas_obj_df_list
            print('Hidden Layer:', str(num_hidden_layer))
            if len(reduced_df_list) > 0:
                temp_list += reduced_df_list
            else:
                print('\tSkip REDUCED*')
            if len(test_df_list) > 0:
                temp_list += test_df_list
            else:
                print('\tSkip TEST*')
            if len(pandas_converted_df_list) > 0:
                temp_list += pandas_converted_df_list
            else:
                print('\tSkip CONVERTED*')
            if len(reduced_df_list) == 0 and len(test_df_list) == 0:
                print('\tNothing to compare')
                continue
            elif len(reduced_df_list) > 0 and len(test_df_list) > 0:
                print('\tCompare All')
            plot_df = pd.concat(temp_list, axis=1)
            for size in SIZE_SET:
                size_columns = list(filter(lambda x: ' ' + str(size) + ' ' in x, plot_df.columns))
                if len(size_columns) <= len(METRIC_SET) + 1:
                    continue
                for metric in METRIC_SET:
                    columns = list(filter(lambda x: metric in x, size_columns))
                    columns.sort(key=sort)
                    ax = plot_df.plot(y=columns, title='Hidden Layer ' + str(num_hidden_layer) + ' ' + metric, xlabel='Epochs', ylabel=metric)
                    box = ax.get_position()
                    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                    # Put a legend to the right of the current axis
                    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='xx-small')
                    if SHOW:
                        plt.show()
                    else:
                        title_portions = [metric, str(size)]
                        plt.savefig('Hidden Layer ' + str(num_hidden_layer) + '\\' + ' '.join(title_portions) + '.png')
                    plt.close('all')

