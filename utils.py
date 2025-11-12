import numpy as np
import torch 
import os
from operator import itemgetter
import pandas as pd

cancer_types = [
    'Breast', 
    'Cervical', 
    'Colon', 
    'Esophageal', 
    'Gastric', 
    'Liver', 
    'Lung', 
    'Pancreatic', 
    'Rectal'
]

type_dict = dict(zip(cancer_types, range(1, len(cancer_types)+1)))

def init_resdict():
    return {
        'auc': [],
        'f1': [],
        'rec': [],
        'prec': [],
        'acc': [],
        'loss': [],
        'loss_val': []
    }


def get_mask(N, L, lengths):
    assert torch.max(lengths) <= L
    mask = torch.zeros((N, L))
    for i, l in enumerate(lengths):
        mask[i, :l] = False
        mask[i, l:] = True
    return mask.bool()


def read_data_file(path, type):
    df = pd.read_csv(path)
    for t in cancer_types:
        if t == type:
            df[t] = [1]*len(df)
        else:
            df[t] = [0]*len(df)
            
    return df


def load_data(path, data_types=[]):
    # load data
    data = load_data_from_dir(path)
    dfs = []
    for data_type in data_types:
        if data_type in data:
            dfs.append(data[data_type])
        else:
            print('data type %s not found in data folder, skip' % data_type)
    data = pd.concat(dfs, axis=0) if len(dfs) > 0 else pd.DataFrame([])
    return data


def load_data_from_dir(path):
    data = {}
    for file in os.listdir(path):
        if file.endswith(".csv"):
            try:
                key = file.split("_")[1]
            except:
                key = file.split('.')[0]
            if not key in cancer_types:
                key = 'other'
            df = read_data_file(os.path.join(path, file), key) 
            if key in cancer_types:   
                df['id'] = reform_id(df['id'].to_numpy(), key)      
            data[key] = df
    return data    


def reform_id(ids, key):
    ids = ids.astype(int)
    if ids[0] > 1e10:
        return ids
    return ids + type_dict[key] * 1e10
    
    
    
def softmax(x, axis=None):
    if axis is None:
        x = np.exp(x) / np.sum(np.exp(x))
    else:
        x = np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)
    return x



def norm_df_ignorena(df: pd.DataFrame, cols, means=None, stds=None):
    
    if means is None or stds is None:
        means = np.zeros(len(cols))
        stds = np.zeros(len(cols))
        for i, col in enumerate(cols):
            mean = df.loc[pd.notnull(df.loc[:, col]), col].mean()
            std = df.loc[pd.notnull(df.loc[:, col]), col].std()
            df.loc[pd.notnull(df.loc[:, col]), col] = (df.loc[pd.notnull(df.loc[:, col]), col] - mean) / std
            means[i] = mean
            stds[i] = std
    else:
        assert len(cols) == len(means), '[Error] given num means doesnot match num features'
        assert len(cols) == len(stds), '[Error] given num stds doesnot match num features'

        for i, col in enumerate(cols):
            df.loc[pd.notnull(df.loc[:, col]), col] = (df.loc[pd.notnull(df.loc[:, col]), col] - means[i]) / stds[i]

    return df, means, stds




def collate_fn(batch, device='cuda'):
    X = torch.from_numpy(np.array([item[0] for item in batch]).astype(np.float32))
    y = torch.from_numpy(np.array([[item[1]] for item in batch]).astype(np.float32))
    
    X = X.to(device)
    y = y.to(device)
    
    try:
        ids = [item[2] for item in batch]
        return X, y, ids
    except:
        pass
    
    return X, y



def split_data(data, train_size=0.8, val_size=0, mode='seq'):
    if mode == 'seq':
        pids = data['id'].unique()
        np.random.shuffle(pids)
        
        train_idx = int(len(pids)*train_size)
        val_idx = int(len(pids)*(train_size+val_size))
        
        train_ids = pids[:train_idx]
        val_ids = pids[train_idx:val_idx]
        
        train_df = data[data['id'].isin(train_ids)]
        val_df = data[data['id'].isin(val_ids)]
        test_df = data[~data['id'].isin(np.concatenate([train_ids, val_ids]))]
    
    elif mode == 'tp':
        train_df = data.sample(frac=train_size)
        test_df = data.drop(train_df.index)
        val_df = None
    
    return train_df, val_df, test_df