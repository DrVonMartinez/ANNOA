import pandas as pd
import random as rnd

from trax.fastmath import numpy as np


def data_generator(batch_input: pd.DataFrame, batch_output: pd.DataFrame, batch_size: int):
    _batch_input = batch_input.reset_index(drop=True)
    _batch_output = batch_output.reset_index(drop=True)
    input1 = []
    output1 = []
    idx = 0
    batch_indexes = [*range(len(_batch_input))]
    rnd.shuffle(batch_indexes)
    while True:
        if idx >= len(_batch_input):
            idx = 0
            rnd.shuffle(batch_indexes)
        idx += 1
        input1.append(np.array(_batch_input.loc[batch_indexes[idx]]))
        # output1.append(np.array(_batch_output.loc[batch_indexes[idx]]))
        temp = np.array(_batch_output.loc[batch_indexes[idx]])
        output1.append((temp * np.array([range(1, 1 + len(_batch_output.columns))])).sum(axis=1).astype(int))
        if len(input1) == batch_size:
            print(np.array(input1).shape, np.array(output1).shape)  # , np.array(output2).shape)
            yield np.array(input1), np.array(output1)  # , np.array(output2)
            # yield np.array(input1).T, np.array(output1).reshape((1, len(output1)))
            input1, output1 = [], []
