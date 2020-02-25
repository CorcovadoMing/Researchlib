import glob
import os
import numpy as np
import pandas as pd
from cv2 import imread
from tqdm.auto import tqdm
from tensorpack.dataflow import *


class DataFlowBuilder(DataFlow):
    def __init__(self, data_sheet):
        super().__init__()
        self.data_sheet = data_sheet
    
    def __len__(self):
        return len(self.data_sheet)
        
    def __iter__(self):
        for path, label in zip(self.data_sheet['Data Path'], self.data_sheet['Remapping Labels']):
            yield imread(path), label
    
    
def Build(name: str, dataset_format: str) -> None:
    support_type = ['lmdb', 'numpy']
    if dataset_format not in support_type:
        raise ValueError(f'Support dataset format is {support_type}')
    
    csv = glob.glob(f'{name}/*.csv')
    phase = list(map(lambda x: x.split('_')[-1].split('.')[0], csv))
    
    for i, j in tqdm(zip(csv, phase), total=len(csv)):
        data_sheet = pd.read_csv(i)
        
        output_file = os.path.join(name, j)
        df = DataFlowBuilder(data_sheet)
        
        if dataset_format == 'lmdb':
            try:
                os.remove(f'{output_file}.lmdb')
                os.remove(f'{output_file}.lmdb-lock')
            except:
                pass
            LMDBSerializer.save(df, f'{output_file}.lmdb')
            