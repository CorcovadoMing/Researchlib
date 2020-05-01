import glob
import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from tensorpack.dataflow import *
from sklearn.utils import shuffle as _shuffle


def _read_encode(path):
    with open(path, 'rb') as f:
        jpeg = f.read()
        jpeg = np.asarray(bytearray(jpeg), dtype='uint8')
        return jpeg


class DataFlowBuilder(DataFlow):
    def __init__(self, data_sheet, mask, bgr2rgb):
        super().__init__()
        self.data_sheet = data_sheet
        self.mask = mask
        self.bgr2rgb = bgr2rgb
    
    def __len__(self):
        return len(self.data_sheet)
        
    def __iter__(self):
        for path, mask, label in zip(self.data_sheet['Data Path'], self.data_sheet['Labels'], self.data_sheet['Remapping Labels']):
            if self.mask:
                x, y = _read_encode(path), _read_encode(mask)
            else:
                x, y = _read_encode(path), label
            
            if self.bgr2rgb:
                x = x[..., ::-1]
                
            yield x, y
    
    
def Build(name: str, dataset_format: str, mask: bool = False, bgr2rgb: bool = False, shuffle: bool = False, split_ratio: float = 0.) -> None:
    '''
        @name: Parser output name
        @dataset_format: one of ['lmdb', 'numpy']
        @mask: True if label is mask or False if is numeric
        @bgr2rgb: Transfer to RGB if raw data is BGR
        @shuffle: Shuffle the dataset
        @split_ratio: split the dataset
    '''
    
    support_type = ['lmdb', 'numpy']
    if dataset_format not in support_type:
        raise ValueError(f'Support dataset format is {support_type}')
    
    csv = glob.glob(f'{name}/parse_result*.csv')
    phase = list(map(lambda x: x.split('_')[-1].split('.')[0], csv))
    
    print(f'Found phase: {phase}')
    if split_ratio > 0:
        print(f'Auto split is enabled, ratio: {split_ratio}')
        original = pd.read_csv(csv[0])
        if shuffle:
            original = _shuffle(original)
            original.reset_index(inplace=True, drop=True)
        new_train = original[:int(split_ratio*len(original))]
        new_val = original[int(split_ratio*len(original)):]
        new_train.to_csv(f'{name}/auto_split_train.csv')
        new_val.to_csv(f'{name}/auto_split_val.csv')
        phase.append('val')
    
    csv = glob.glob(f'{name}/auto_split*.csv')
    phase = list(map(lambda x: x.split('_')[-1].split('.')[0], csv))
    
    print(f'Found phase: {phase}')
    
    for i, j in tqdm(zip(csv, phase), total=len(csv)):
        data_sheet = pd.read_csv(i)
        
        if shuffle:
            data_sheet = _shuffle(data_sheet)
            data_sheet.reset_index(inplace=True, drop=True)
        
        output_file = os.path.join(name, j)
        df = DataFlowBuilder(data_sheet, mask, bgr2rgb)
        
        if dataset_format == 'lmdb':
            try:
                os.remove(f'{output_file}.lmdb')
                os.remove(f'{output_file}.lmdb-lock')
            except:
                pass
            LMDBSerializer.save(df, f'{output_file}.lmdb')
            