import glob
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from Utilities.Constants import SIZE_SET, SHOW
from Utilities.Tensor_Constants import NUM_EPOCHS, NUM_HIDDEN_LAYERS, EXPANDED_METRIC_SET


def sort(value):
    return int((value.split(',')[1]).split(' ')[-1])


def plot(data_df, current_metric, current_title, class_count=None, data_count=None):
    ax = data_df.plot(xlabel='Epochs', ylabel=current_metric)
    plt.suptitle(current_title, y=1.05, fontsize=18)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='xx-small')
    if SHOW:
        plt.show()
    elif class_count is None:
        plt.title(data_count + ' ' + current_metric, fontsize=10)
        plt.savefig('Same Data\\' + current_metric + '\\' + data_count + ' ' + current_title + '.png')
    elif data_count is None:
        plt.title(class_count + ' ' + current_metric, fontsize=10)
        plt.savefig('Same Classes\\' + current_metric + '\\' + class_count + ' ' + current_title + '.png')
    else:
        plt.title(current_metric, fontsize=10)
        plt.savefig(data_count + '\\' + class_count + '\\' + current_metric + '\\' + current_title + '.png')
    plt.close('all')


cwd = os.getcwd()
x = range(1, NUM_EPOCHS + 1)
for class_count_var in ['Full Classes', 'Limited Classes']:
    for data_count_var in ['All Data', 'Partial Data']:
        os.chdir(cwd)
        os.chdir('..\\Results\\' + data_count_var + '\\' + class_count_var + '\\')
        print(os.getcwd())
        # Build DataFrame
        oa_hidden_layer_list = []
        for hidden_layer in NUM_HIDDEN_LAYERS:
            hidden_layer_files = list(filter(lambda data: data_count_var not in data, glob.glob('PD*Hidden Layer ' + str(hidden_layer) + '*.csv')))
            if len(hidden_layer_files) == 0:
                continue
            hidden_layer_files.sort(key=sort)
            for file in hidden_layer_files:
                size = file.split(',')[1].split(' ')[-1]
                if int(size) < 100:
                    size = '0' + size
                temp = pd.read_csv(file)
                columns = [col + ' HL ' + str(hidden_layer) + ' S ' + size + '.' for col in temp.columns]
                oa_hidden_layer_list.append(temp.set_axis(columns, axis=1))
        if len(oa_hidden_layer_list) == 0:
            print(list(filter(lambda data: data_count_var not in data, glob.glob('PD*.csv'))))
            continue
        else:
            print(str(len(oa_hidden_layer_list)), 'Files Loaded')
        oa_hidden_layer_df = pd.concat(oa_hidden_layer_list, axis=1)
        oa_hidden_layer_df['Index'] = x
        oa_hidden_layer_df = oa_hidden_layer_df.set_index('Index')
        oa_hidden_layer_df = oa_hidden_layer_df.reindex(sorted(list(oa_hidden_layer_df.columns)), axis=1)
        os.chdir(cwd)
        os.chdir('\\')
        for metric in sorted(EXPANDED_METRIC_SET):
            # Compare All together
            current_metric_df = oa_hidden_layer_df[filter(lambda layers: metric in layers, oa_hidden_layer_df.columns)]
            if current_metric_df.empty:
                continue
            compare_all_title = 'Compare All Information'
            plot(current_metric_df, metric, compare_all_title, class_count=class_count_var, data_count=data_count_var)

            # Compare Across Sizes
            for hidden_layer in NUM_HIDDEN_LAYERS:
                hidden_layer_df = current_metric_df[filter(lambda layers: 'HL ' + str(hidden_layer) + ' ' in layers, current_metric_df.columns)]
                if hidden_layer_df.empty:
                    continue
                hidden_layer_title = 'Compare Sizes with Hidden Layer ' + str(hidden_layer)
                plot(hidden_layer_df, metric, hidden_layer_title, class_count_var, data_count_var)

            # Compare Across Hidden Layer Counts
            for size in SIZE_SET:
                if size < 100:
                    str_size = '0' + str(size)
                else:
                    str_size = str(size)
                size_df = current_metric_df[filter(lambda layers: 'S ' + str_size + '.' in layers, current_metric_df.columns)]
                if size_df.empty:
                    continue
                size_title = 'Compare Hidden Layers with Size ' + str(size)
                plot(size_df, metric, size_title, class_count_var, data_count_var)

# Same Data
for class_count_var in ['Full Classes', 'Limited Classes']:
    os.chdir(cwd)
    os.chdir('../Results\\')
    print(os.getcwd() + '\\*\\' + class_count_var)
    # Build DataFrame
    oa_hidden_layer_list = []
    for hidden_layer in NUM_HIDDEN_LAYERS:
        hidden_layer_files = list(glob.glob('*\\' + class_count_var + '\\PD*Hidden Layer ' + str(hidden_layer) + '*.csv', recursive=True))
        if len(hidden_layer_files) == 0:
            continue
        hidden_layer_files.sort(key=sort)
        for file in hidden_layer_files:
            if 'All Data' in file:
                style = 'All Data'
            else:
                style = 'Partial Data'
            size = file.split(',')[1].split(' ')[-1]
            if int(size) < 100:
                size = '0' + size
            temp = pd.read_csv(file)
            columns = [style + ' ' + col + ' HL ' + str(hidden_layer) + ' S ' + size + '.' for col in temp.columns]
            oa_hidden_layer_list.append(temp.set_axis(columns, axis=1))
    print(str(len(oa_hidden_layer_list)), 'Files Loaded')
    oa_hidden_layer_df = pd.concat(oa_hidden_layer_list, axis=1)
    oa_hidden_layer_df['Index'] = x
    oa_hidden_layer_df = oa_hidden_layer_df.set_index('Index')
    oa_hidden_layer_df = oa_hidden_layer_df.reindex(sorted(list(oa_hidden_layer_df.columns)), axis=1)
    os.chdir(cwd)
    os.chdir('\\')
    for metric in EXPANDED_METRIC_SET:
        # Compare All together
        current_metric_df = oa_hidden_layer_df[filter(lambda layers: metric in layers, oa_hidden_layer_df.columns)]
        if current_metric_df.empty:
            continue
        compare_all_title = 'Compare All Information'
        plot(current_metric_df, metric, compare_all_title, class_count_var)

        # Compare Across Hidden Layer Counts
        for size in SIZE_SET:
            if size < 100:
                str_size = '0' + str(size)
            else:
                str_size = str(size)
            size_df = current_metric_df[filter(lambda layers: 'S ' + str_size + '.' in layers, current_metric_df.columns)]
            if size_df.empty:
                continue
            size_title = 'Compare Data Hidden Layers with Size ' + str(size)
            plot(size_df, metric, size_title, class_count_var)

        # Compare Across Sizes
        for hidden_layer in NUM_HIDDEN_LAYERS:
            hidden_layer_df = current_metric_df[filter(lambda layers: 'HL ' + str(hidden_layer) + ' ' in layers, current_metric_df.columns)]
            if hidden_layer_df.empty:
                continue
            hidden_layer_title = 'Compare Data Sizes with Hidden Layer ' + str(hidden_layer)
            plot(hidden_layer_df, metric, hidden_layer_title, class_count_var)

# Same Classes
for data_count_var in ['All Data', 'Partial Data']:
    os.chdir(cwd)
    os.chdir('..\\Results\\' + data_count_var + '\\')
    print(os.getcwd() + '\\*\\')
    # Build DataFrame
    oa_hidden_layer_list = []
    for hidden_layer in NUM_HIDDEN_LAYERS:
        hidden_layer_files = list(glob.glob('*\\PD*Hidden Layer ' + str(hidden_layer) + '*.csv'))
        assert len(hidden_layer_files) > 0
        hidden_layer_files.sort(key=sort)
        for file in hidden_layer_files:
            size = file.split(',')[1].split(' ')[-1]
            temp = pd.read_csv(file)
            columns = [data_count_var + ' ' + col + ' HL ' + str(hidden_layer) + ' S ' + size + '.' for col in temp.columns]
            oa_hidden_layer_list.append(temp.set_axis(columns, axis=1))
    print(str(len(oa_hidden_layer_list)), 'Files Loaded')
    oa_hidden_layer_df = pd.concat(oa_hidden_layer_list, axis=1)
    oa_hidden_layer_df['Index'] = x
    oa_hidden_layer_df = oa_hidden_layer_df.set_index('Index')
    oa_hidden_layer_df = oa_hidden_layer_df.reindex(sorted(list(oa_hidden_layer_df.columns)), axis=1)
    os.chdir(cwd)
    os.chdir('\\')
    for metric in sorted(EXPANDED_METRIC_SET):
        # Compare All together
        current_metric_df = oa_hidden_layer_df[filter(lambda layers: metric in layers, oa_hidden_layer_df.columns)]
        if current_metric_df.empty:
            continue
        compare_all_title = 'Compare All Information'
        plot(current_metric_df, metric, compare_all_title, None, data_count_var)

        # Compare Across Hidden Layer Counts
        for size in SIZE_SET:
            size_df = current_metric_df[filter(lambda layers: 'S ' + str_size + '.' in layers, current_metric_df.columns)]
            if size_df.empty:
                continue
            size_title = 'Compare Class Hidden Layers with Size ' + str(size)
            plot(size_df, metric, size_title, None, data_count_var)

        # Compare Across Sizes
        for hidden_layer in NUM_HIDDEN_LAYERS:
            hidden_layer_df = current_metric_df[filter(lambda layers: 'HL ' + str(hidden_layer) + ' ' in layers, current_metric_df.columns)]
            if hidden_layer_df.empty:
                continue
            hidden_layer_title = 'Compare Class Sizes with Hidden Layer ' + str(hidden_layer)
            plot(hidden_layer_df, metric, hidden_layer_title, None, data_count_var)
