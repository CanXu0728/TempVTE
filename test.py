import os
import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, accuracy_score
from utils import load_data, norm_df_ignorena, collate_fn, init_resdict
from datasets.dataset import VTEDataset
from LR import LR
from torch.utils.data import DataLoader
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_path', type=str, required=True, help="path to model checkpoints")
    parser.add_argument('--config', type=str, required=True, help='path to config file')
    parser.add_argument('--device', type=int, default=0, help='cuda device number, set to -1 if using cpu')
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--infer', action="store_true", default=False, help='if save results')

    args = parser.parse_args()
    return args


def test(args):
    device = torch.device('cuda:%d'%args.device if args.device>=0 and torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)

    with open(args.config, 'r') as f:
        cfg = json.loads(f.read()) 
        
    # read data
    if isinstance(cfg['data_type'], str):
        data_types = [cfg['data_type']]
    else:
        data_types = cfg['data_type']
    data = load_data(cfg['data_folder'], data_types)

    feature_cols = cfg["feature_cols"]
    time_col = cfg["time_col"]
    label_col = cfg["label_col"]

    # normalize        
    if 'means' in cfg and 'stds' in cfg:
        data, means, stds = norm_df_ignorena(data, feature_cols, cfg['means'], cfg['stds'])
    else:
        data, means, stds = norm_df_ignorena(data, feature_cols)
        cfg['means'] = means.tolist()
        cfg['stds'] = stds.tolist()

    # build dataset
    if args.infer:
        test_dataset = VTEDataset(data, feature_cols, time_col, label_col, infer=True)
    else:    
        test_dataset = VTEDataset(data, feature_cols, time_col, label_col, test=True)
    test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), collate_fn=lambda x: collate_fn(x, device=device))

    # load model
    model = LR(3*len(feature_cols))
    model.to(device)
    model.load_state_dict(torch.load(args.model_path, weights_only=True))

    # test
    preds =[]
    ys = []
    ids = []
    for X in test_dataloader:
        if args.infer:
            X, y, pids = X
        else:
            X, y = X
            pids = []
        pred = model(X)        
        pred = pred.flatten().cpu().detach().numpy()
        y = y.flatten().cpu().detach().numpy()
        
        preds.append(pred)
        ys.append(y)
        ids += pids
    preds = np.hstack(preds)
    ys = np.hstack(ys)

    # if infer, save results
    if args.infer:
        df = pd.DataFrame(zip(ids, ys, preds), columns=['id', 'y', 'pred'])
        df.to_csv(os.path.join(args.save_dir, 'test_res.csv'), index=False)
        print('results saved at ', os.path.join(args.save_dir, 'test_res.csv'))

    # compute metrics and visualize
    auc = roc_auc_score(y, pred)
    pred = np.where(pred < 0.5, 0, 1)
    f1, rec, prec, acc = f1_score(y, pred), recall_score(y, pred), precision_score(y, pred), accuracy_score(y, pred)
    print('============================================')
    print('test auc: %.4f' % auc)
    print('test acc: %.4f' % acc)
    print('test f1: %.4f' % f1)
    print('test rec: %.4f' % rec)
    print('test prec: %.4f' % prec)



if __name__ == '__main__':
    args = parse_args()
    test(args)