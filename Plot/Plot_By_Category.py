import glob

import matplotlib.pyplot as plt
import pandas as pd
from natsort import natsorted

import Utilities.Determine_Metrics as dm
from Constants.Constants import SIZE_SET, DIMENSION_SET
from Constants.Expanded_Constants import NUM_HIDDEN_LAYERS, NUM_EPOCHS, REFERENCE_LIST
from Constants.Storage_Constants import RESULT_PATH, PLOT_PATH

SELECTED_REFERENCES = REFERENCE_LIST
x = range(1, NUM_EPOCHS + 1)
if len(SELECTED_REFERENCES) > 2:
    dist_add_on = 'Expanded Dist Set '
else:
    dist_add_on = ''


def sort(value: str):
    start = value.index('HL')-1
    return value[start:]


def sort2(value: str):
    return int(value.split(' ')[-3])


def rename_columns(data_frame: pd.DataFrame) -> pd.DataFrame:
    columns = data_frame.columns
    new_columns = []
    for col in sorted(columns, key=sort):
        if 'S 0' in str(col):
            new_columns.append(str(col).replace('S 0', 'S '))
        else:
            new_columns.append(str(col))
    # new_columns = sorted(new_columns, key=sort)
    data_frame.set_axis(new_columns, axis=1, inplace=True)
    return data_frame


def name_columns(file_path: str):
    split_file_path = file_path.split('\\')
    data_style = split_file_path[3]
    class_style = split_file_path[4]
    file_title = (split_file_path[-1]).split(',')
    ref_style = file_title[0]
    if ' PCA' == file_title[3]:
        ref_style += ' PCA'
    if '[' in file_path:
        size: str = 'All-Sizes'
    else:
        size: str = file_title[-2].split(' ')[-1]
        size = 'S ' + size
    hidden_layer: str = file_title[2].split(' ')[-1]
    dim: str = file_title[1].split(' ')[-1]
    return ref_style, dim, data_style, hidden_layer, size, class_style


def load_data() -> pd.DataFrame:
    all_files = []
    for ref in SELECTED_REFERENCES:
        for data_count_var in ['All Data', 'Partial Data']:
            for class_count_var in ['Full Classes', 'Limited Classes']:
                glob_files = list(filter(lambda z: '[' not in z, glob.glob(RESULT_PATH.format(reference=ref, data=data_count_var, classes=class_count_var) + '*.csv')))
                for glob_file in glob_files:
                    assert glob_file not in all_files
                all_files += glob_files
    oa_df_list = []
    for file in all_files:
        ref_style, dim, data_style, hidden_layer, size, class_style = name_columns(file)
        if dim == '3':
            continue
        temp = pd.read_csv(file)
        if 'False Negatives' in temp.columns:
            temp = dm.false_positive_rate(temp)
            temp = dm.f1_score(temp)
            temp = dm.specificity(temp)
            temp = dm.false_omission_rate(temp)
            temp.drop(columns=['True Positives', 'True Negatives', 'False Positives', 'False Negatives'], inplace=True)
        columns = [ref_style + ' D ' + dim + ' ' + data_style + ' ' + col + ' HL ' + hidden_layer + ' ' + size + ' ' + class_style for col in temp.columns]
        oa_df_list.append(temp.set_axis(columns, axis=1))
    return pd.concat(oa_df_list, axis=1)


def plot(data_df, current_metric, current_title, class_count=None, data_count=None, show=False):
    data_df = rename_columns(data_df)
    ax = data_df.plot(xlabel='Epochs', ylabel=current_metric)
    plt.suptitle(current_title, y=1.05, fontsize=18)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='xx-small')
    if class_count is None:
        plt.title(data_count + ' ' + current_metric, fontsize=10)
        path_title = 'Same Data\\' + current_metric + '\\' + dist_add_on + data_count + ' ' + current_title + '.png'
    elif data_count is None:
        plt.title(class_count + ' ' + current_metric, fontsize=10)
        path_title = 'Same Classes\\' + current_metric + '\\' + dist_add_on + class_count + ' ' + current_title + '.png'
    else:
        plt.title(current_title, fontsize=10)
        path_title = data_count + '\\' + class_count + '\\' + current_metric + '\\' + dist_add_on + current_title + '.png'

    if show:
        plt.show()
    else:
        plt.savefig(PLOT_PATH + path_title)
    plt.close('all')


def plot_by_pca(metric_df: pd.DataFrame, title: str, metric: str, class_count_var: str, data_count_var: str, show: bool):
    pca_df = metric_df[filter(lambda layers: 'PCA' in layers, metric_df.columns)]
    pca_title = title + ' and PCA'
    if not pca_df.empty:
        plot(pca_df, metric, pca_title, class_count_var, data_count_var, show)

    no_pca_df = metric_df[filter(lambda layers: 'PCA' not in layers, metric_df.columns)]
    no_pca_title = title + ' without PCA'
    if not no_pca_df.empty:
        plot(no_pca_df, metric, no_pca_title, class_count_var, data_count_var, show)


def plot_by_size(metric_df: pd.DataFrame, size: str, title: str, metric: str, class_count_var: str, data_count_var: str, show: bool):
    size_df = metric_df[filter(lambda layers: ' 0' + size + ' ' in layers or ' ' + size + ' ' in layers or 'All-Sizes' in layers, metric_df.columns)]
    if size_df.empty:
        return
    size_title = title + ' of Hidden Layers with Size ' + str(size)
    plot_by_pca(size_df, size_title, metric, class_count_var, data_count_var, show)


def plot_by_hidden_layer(metric_df: pd.DataFrame, hidden_layer: int, title: str, metric: str, class_count_var: str, data_count_var: str, show: bool):
    hidden_layer_df = metric_df[filter(lambda layers: 'HL ' + str(hidden_layer) + ' ' in layers, metric_df.columns)]
    if hidden_layer_df.empty:
        return
    hidden_layer_title = title + ' of Sizes with Hidden Layer ' + str(hidden_layer)
    plot_by_pca(hidden_layer_df, hidden_layer_title, metric, class_count_var, data_count_var, show)


def plot_by_dimension(metric_df: pd.DataFrame, dimension: int, metric: str, class_count_var: str, data_count_var: str, show: bool):
    dim_df = metric_df[filter(lambda layers: 'D ' + str(dimension) + ' ' in layers, metric_df.columns)]
    if dim_df.empty:
        return
    for size in SIZE_SET:
        plot_by_size(dim_df, str(size), 'Compare Dim ' + str(dimension) + ' ' + metric, metric, class_count_var, data_count_var, show)
    for hidden_layer in NUM_HIDDEN_LAYERS:
        plot_by_hidden_layer(dim_df, hidden_layer, 'Compare Dim ' + str(dimension) + ' ' + metric, metric, class_count_var, data_count_var, show)


def group_by_category(oa_category_df: pd.DataFrame, class_count_var=None, data_count_var=None, show=False):
    # Build DataFrame
    if not class_count_var and not data_count_var:
        raise ValueError('There must be at least either Same Data or Same Class')
    oa_category_df = oa_category_df.reindex(sorted(list(oa_category_df.columns), key=sort2), axis=1)
    for metric in sorted(['Accuracy', 'Loss', 'Mean Absolute Error', 'Precision', 'Recall', 'False Positive Rate', 'F1 Score', 'Specificity', 'False Omission Rate']):
        current_metric_df = oa_category_df[filter(lambda layers: metric in layers, oa_category_df.columns)]
        for dim in DIMENSION_SET:
            plot_by_dimension(current_metric_df, dim, metric, class_count_var, data_count_var, show)
        for size in SIZE_SET:
            plot_by_size(current_metric_df, str(size), 'Compare ' + metric, metric, class_count_var, data_count_var, show)
        for hidden_layer in NUM_HIDDEN_LAYERS:
            plot_by_hidden_layer(current_metric_df, hidden_layer, 'Compare ' + metric, metric, class_count_var, data_count_var, show)


df = load_data()
# print('\n'.join(natsorted(df.columns)))
df = df.reindex(natsorted(df.columns), axis=1)
df.info()
for data_var in ['All Data', 'Partial Data']:
    for class_var in ['Full Classes', 'Limited Classes']:
        print(data_var, class_var, sep='\t')
        group_by_category(df[filter(lambda layers: data_var in layers and class_var in layers, df.columns)], class_count_var=class_var, data_count_var=data_var)
for class_var in ['Full Classes', 'Limited Classes']:
    data_var = None
    print(data_var, class_var, sep='\t')
    group_by_category(df[filter(lambda layers: class_var in layers, df.columns)], class_count_var=class_var, data_count_var=data_var)
for data_var in ['All Data', 'Partial Data']:
    class_var = None
    print(data_var, class_var, sep='\t')
    group_by_category(df[filter(lambda layers: data_var in layers, df.columns)], class_count_var=class_var, data_count_var=data_var)
