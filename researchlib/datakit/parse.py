import os
import numpy as np
import glob
import re
import pandas as pd
from tqdm.auto import tqdm
import pickle
import shutil


def parsing_path(path, parse_format):
    glob_pattern = os.path.join(path,
                                parse_format.replace('{label}', '*').replace('{data}', '*')
                               )
    match_pattern = os.path.join(path,
                                 parse_format.replace('{label}', '(.+)').replace('{data}', '(.+)')
                                ).replace('*', '.+')
    data_position = parse_format.find('{data}')
    label_position = parse_format.find('{label}')
    return data_position, label_position, glob_pattern, match_pattern



def Parse(name: str, path: str, parse_format: str, phase: str, label_mapping: str = None, label_mapping_sep: str = None, map_file: bool = False, force: bool = False) -> None:
    if force:
        shutil.rmtree(name, ignore_errors = True)
    os.makedirs(name, exist_ok = True)
    
    if ',' in parse_format:
        data_format, label_format = parse_format.split(',')
        data_format, label_format = data_format.strip(), label_format.strip()
        data_position, _, data_glob_pattern, data_match_pattern = parsing_path(path, data_format)
        _, label_position, label_glob_pattern, label_match_pattern = parsing_path(path, label_format)
    else:
        data_position, label_position, glob_pattern, match_pattern = parsing_path(path, parse_format)
        data_glob_patten, label_glob_patten = glob_pattern
        data_match_pattern, label_match_pattern = match_pattern
        
    print(data_position, label_position)
    print(data_glob_pattern, label_glob_pattern)
    print(data_match_pattern, label_match_pattern)

    # data could be left blank, but label should be given
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

            
    data_found = np.array(glob.glob(data_glob_pattern))
    label_found = np.array(glob.glob(label_glob_pattern))
    print(f'Found {len(data_found)} data, {len(label_found)} labels')

    # Deal with label mapping
    if label_mapping is not None:
        mapping_df = pd.read_csv(os.path.join(path, label_mapping), sep = label_mapping_sep, header = None)
        mapping_dict = {i: j for i, j in zip(mapping_df[0].values, mapping_df[1].values)}

    data_path = []
    label = []
    if map_file:
        data_match_thing = np.array([re.findall(data_match_pattern, i) for i in data_found]).flatten()
        label_match_thing = np.array([re.findall(label_match_pattern, i) for i in label_found]).flatten()
        _, index1, index2 = np.intersect1d(data_match_thing, label_match_thing, return_indices=True)
        data_found = data_found[index1]
        label_found = label_found[index2]
        print(f'Found {len(data_found)} data, {len(label_found)} labels after intersection')
        
    for i, j in tqdm(zip(data_found, label_found)):
        data_match = re.match(data_match_pattern, i)
        label_match = re.match(label_match_pattern, j)

        data_path.append(data_match.group())
        label_raw = label_match.group()
        try:
            label.append(mapping_dict[label_raw])
        except:
            label.append(label_raw)

    print(f'Successfully parse {len(data_path)} data with {len(np.unique(label))} labels')

    print(f'Assign numeric labels ...')
    remap_file = os.path.join(name, 'remap.pickle')
    is_remap_file_exists = os.path.exists(remap_file)
    if is_remap_file_exists:
        print('Found remapping exists')
        with open(remap_file, 'rb') as f:
            remap = pickle.load(f)
    else:
        print('Build remapping files')
        remap = {j: i for i, j in enumerate(np.unique(label))}
        with open(remap_file, 'wb') as f:
            pickle.dump(remap, f)
    numeric_label = [remap[i] for i in label]
        

    output_path = os.path.join(name, f'parse_result_{phase}.csv')
    print(f'Write to {output_path}')
    df = pd.DataFrame(list(zip(data_path, label, numeric_label)))
    df.columns = ['Data Path', 'Labels', 'Remapping Labels']
    df.to_csv(output_path, index = False)

    print(f'Done.')