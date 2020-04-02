import glob
import os
import numpy as np
import pandas as pd
from cv2 import imread
from tqdm.auto import tqdm
from tensorpack.dataflow import *


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
                x, y = imread(path), imread(mask)
            else:
                x, y = imread(path), label
            
            if self.bgr2rgb:
                x = x[..., ::-1]
                
            yield x, y
    
    
def Build(name: str, dataset_format: str, mask: bool = False, bgr2rgb: bool = False) -> None:
    '''
        @name: Parser output name
        @dataset_format: one of ['lmdb', 'numpy']
        @mask: True if label is mask or False if is numeric
        @bgr2rgb: Transfer to RGB if raw data is BGR
    '''
    
    support_type = ['lmdb', 'numpy']
    if dataset_format not in support_type:
        raise ValueError(f'Support dataset format is {support_type}')
    
    csv = glob.glob(f'{name}/*.csv')
    phase = list(map(lambda x: x.split('_')[-1].split('.')[0], csv))
    
    for i, j in tqdm(zip(csv, phase), total=len(csv)):
        data_sheet = pd.read_csv(i)
        
        output_file = os.path.join(name, j)
        df = DataFlowBuilder(data_sheet, mask, bgr2rgb)
        
        if dataset_format == 'lmdb':
            try:
                os.remove(f'{output_file}.lmdb')
                os.remove(f'{output_file}.lmdb-lock')
            except:
                pass
            LMDBSerializer.save(df, f'{output_file}.lmdb')
            