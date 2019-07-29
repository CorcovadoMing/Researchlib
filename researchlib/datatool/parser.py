import os
import numpy as np
import glob
import re
from imageio import imread
import pandas as pd
from tqdm.auto import tqdm
from ..utils import ParallelExecutor
import torch.multiprocessing as mp
import cv2
import math
import zarr


class _Parser:
    def __init__(self):
        pass

    
    def parse(self, name: str, path: str, parse_format: str, label_mapping: str = None, sep: str = None) -> None:
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
        
        
        # Deal with label mapping
        if label_mapping is not None:
            mapping_df = pd.read_csv(os.path.join(path, label_mapping), sep=sep, header=None)
            mapping_dict = {i: j for i, j in zip(mapping_df[0].values, mapping_df[1].values)}
        

        data_path = []
        label = []
        for i in tqdm(found):
            m = re.match(match_pattern, i)
            data_path.append(m.group(data_idx))
            label_raw = m.group(label_idx)
            try:
                label.append(mapping_dict[label_raw])
            except:
                label.append(label_raw)
        
        df = pd.DataFrame(list(zip(data_path, label)))
        df.columns = ['Data Path', 'Labels']
        df.to_csv(os.path.join(name, 'parse_result.csv'), index=False)
    
    
    def build(self, name: str, shape, batch_size: int = 1000, num_workers: int = os.cpu_count(), force: bool = False):
        def _task(*args):
            # TODO: shape needs to be passed from outside
            desc = args[0]
            data_zarr = args[1]
            label_zarr = args[2]
            data_zarr[desc['id']] = cv2.resize(imread(desc['data'], pilmode='RGB'), (256, 256)).transpose((2,0,1))
            label_zarr[desc['id']] = int(desc['label'])
            return desc['id']
        
        if os.path.exists(os.path.join(name, 'db.zarr')) and force != True:
            print('There already exists a database, you can set `force=True` to overwrite the database')
        else:
            parse_file = pd.read_csv(os.path.join(name, 'parse_result.csv'))
            parse_iter = zip(parse_file['Data Path'].values, parse_file['Labels'].values)
            total_length = len(parse_file['Data Path'].values)
            total_epoch = math.ceil(total_length/batch_size)

            # Initialization
            root = zarr.open(os.path.join(name, 'db.zarr'), mode='w')
            data_zarr = root.zeros('data', shape=(1, *shape), chunks=(1, *shape), dtype='f', compressor=None)
            label_zarr = root.zeros('label', shape=(1,), chunks=(1,), dtype='i', compressor=None)

            # resize the storage in the first place
            data_zarr.resize(total_length, *shape)
            label_zarr.resize(total_length,)

            executor = ParallelExecutor(_task, max_job=batch_size, num_workers=num_workers)
            executor.start(data_zarr, label_zarr)

            try:
                executor.start(data_zarr, label_zarr)
                for i in tqdm(range(total_epoch)):
                    for j in range(batch_size):
                        try:
                            data, label = next(parse_iter)
                            executor.put({'id': (i * batch_size) + j, 'data': data, 'label': label})
                        except:
                            continue
                    _ = executor.wait()
            except:
                pass
            finally:
                executor.stop(data_zarr, label_zarr)
