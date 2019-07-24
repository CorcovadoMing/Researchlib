import os
import numpy as np
import glob
import re
from imageio import imread
import pandas as pd


class _Builder:
    def __init__(self):
        pass

    def build(self, name: str, path: str, parse_format: str) -> None:
        os.makedirs(name, exist_ok=True)

        glob_pattern = os.path.join(
            path,
            parse_format.replace('{label}', '*').replace('{data}', '*'))
        match_pattern = os.path.join(
            path,
            parse_format.replace('{label}',
                                 '(.+)').replace('{data}',
                                                 '(.+)')).replace('*', '.+')

        data_position = parse_format.find('{data}')
        label_position = parse_format.find('{label}')

        # data could be left blank, but label should give
        assert label_position >= 0

        if data_position < label_position:
            data_idx, label_idx = 1, 2
            if data_position == -1:
                data_idx -= 1
                label_idx -= 1
        else:
            data_idx, label_idx = 2, 1
            if label_position == -1:
                data_idx -= 1
                label_idx -= 1

        found = glob.glob(glob_pattern)

        print(f'Found {len(found)} data')

        data_path = []
        label = []
        for i in found:
            m = re.match(match_pattern, i)
            data_path.append(m.group(data_idx))
            label.append(m.group(label_idx))

        df = pd.DataFrame(list(zip(data_path, label)))
        df.columns = ['Data Path', 'Labels']
        df.to_csv(os.path.join(name, 'parse_result.csv'), index=False)
