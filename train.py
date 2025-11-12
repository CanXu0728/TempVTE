import os
import torch
import json
import time
import numpy as np
import argparse
from utils import load_data, norm_df_ignorena, collate_fn, init_resdict, split_data
from datasets.dataset import VTEDataset
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, accuracy_score
from LR import LR
from torch.utils.data import WeightedRandomSampler, DataLoader


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--config', type=str, required=True, help='Specify config file corresponding to the selected model')
    parser.add_argument('--device', type=int, default=0, help='cuda device number, set to -1 if using cpu')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')

    args = parser.parse_args()
    return args


def train(args):
    
    ## init cuda device and save path
    device = torch.device('cuda:%d'%args.device if args.device>=0 and torch.cuda.is_available() else 'cpu')
    stamp = time.strftime('%Y%m%d.%H%M%S', time.localtime(time.time()))
    save_dir = os.path.join(args.save_dir, stamp)
    os.makedirs(save_dir)
    
    ## load config and data
    with open(args.config, 'r') as f:
        cfg = json.loads(f.read()) 
    if isinstance(cfg['data_type'], str):
        data_types = [cfg['data_type']]
    else:
        data_types = cfg['data_type']
    feature_cols = cfg["feature_cols"]
    time_col = cfg["time_col"]
    label_col = cfg["label_col"]
    
    ## normalization
    data = load_data(cfg['data_folder'], data_types)
    if 'means' in cfg and 'stds' in cfg:
        data, means, stds = norm_df_ignorena(data, feature_cols, cfg['means'], cfg['stds'])
    else:
        data, means, stds = norm_df_ignorena(data, feature_cols)
        cfg['means'] = means.tolist()
        cfg['stds'] = stds.tolist()

    # save config file
    with open(os.path.join(save_dir, os.path.basename(args.config)), 'w+', encoding='utf8') as f:
        f.write(json.dumps(cfg, ensure_ascii=False))
    
    # train
    res = init_resdict()
    val_res = init_resdict()
    for step in range(cfg["step"]):
        os.makedirs(os.path.join(save_dir, str(step)))
        
        ## build model
        model = LR(3*len(feature_cols))
        model.to(device)
        
        ## data split
        # if using validation set ===========================================================
        # train_data, val_data, test_data = split_data(data, train_size=0.7, val_size=0.15)
        # val_dataset = VTEDataset(val_data, feature_cols, time_col, label_col, test=True)
        # val_dataloader = DataLoader(val_dataset, batch_size=len(val_dataset), collate_fn=lambda x: collate_fn(x, device=device))
        # ===================================================================================
        train_data, _, test_data = split_data(data)
        
        ## build dataset
        train_dataset = VTEDataset(train_data, feature_cols, time_col, label_col)
        train_vte = np.sum(train_dataset.y)
        train_weights = np.where(np.array(train_dataset.y) == 1, (len(train_dataset.y)-train_vte)/train_vte, 1)
        train_sampler = WeightedRandomSampler(train_weights, len(train_dataset.y), replacement=True) ## balance sample size of classes
        train_dataloader = DataLoader(train_dataset, batch_size=cfg["batch_size"], sampler=train_sampler, collate_fn=lambda x: collate_fn(x, device=device))
        test_dataset = VTEDataset(test_data, feature_cols, time_col, label_col, test=True)
        test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), collate_fn=lambda x: collate_fn(x, device=device))
        
        # defining the optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg["lr"])
        # defining Cross-Entropy loss
        criterion = torch.nn.functional.binary_cross_entropy
        
        # train
        best_score = 0
        for epoch in range(cfg["epochs"]):
            model.train()
            for i, (X, y) in enumerate(train_dataloader):
                optimizer.zero_grad()
                outputs = model(X)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
            res['loss'].append(loss.item())
            
            ### if use validation ===================================================================
            # auc, acc, f1, rec, prec, val_loss = test_model(model, val_dataloader, criterion)
            ### =====================================================================================
            auc, acc, f1, rec, prec, val_loss = test_model(model, test_dataloader, criterion)
            res['loss_val'].append(val_loss)
            print('Epoch %d: train loss: %.4f val_loss: %.4f || auc: %.4f accuracy: %.4f f1: %.4f recall: %.4f precision: %.4f' 
                    % (epoch, loss.item(), val_loss, auc, f1, rec, prec, acc))

            # save if better than current best model
            score = np.mean([auc, f1, acc]) # model selection score
            if score > best_score:
                best_score = score
                torch.save(model.state_dict(), os.path.join(save_dir, str(step), './best.pt'))  
                print('score %.4f best model saved' % score)
            
            # save at fixed epoch
            if (epoch+1) % 100 == 0:
                torch.save(model.state_dict(), os.path.join(save_dir, str(step), f'./epoch_{epoch}.pt'))  
        
        model.load_state_dict(torch.load(os.path.join(save_dir, str(step), './best.pt'), weights_only=True))
        auc, acc, f1, rec, prec, val_loss = test_model(model, test_dataloader, criterion)
        res['auc'].append(auc)
        res['f1'].append(f1)
        res['rec'].append(rec)
        res['prec'].append(prec)
        res['acc'].append(acc)
        
        ## print results with confidence intervals
        print('TempVTE: ')
        ### if use validation ===========================================================================================
        # auc, acc, f1, rec, prec, val_loss = test_model(model, val_dataloader, criterion)
        # val_res['auc'].append(auc)
        # val_res['f1'].append(f1)
        # val_res['rec'].append(rec)
        # val_res['prec'].append(prec)
        # val_res['acc'].append(acc)
        # print('val auc: %.4f (%.4f)' % (np.mean(val_res['auc']), 1.96/np.sqrt(cfg["step"]) * np.std(val_res['auc'])))
        # print('val accuracy: %.4f (%.4f)' % (np.mean(val_res['acc']), 1.96/np.sqrt(cfg["step"]) * np.std(val_res['acc'])))
        # print('val f1 score: %.4f (%.4f)' % (np.mean(val_res['f1']), 1.96/np.sqrt(cfg["step"]) * np.std(val_res['f1'])))
        # print('val recall: %.4f (%.4f)' % (np.mean(val_res['rec']), 1.96/np.sqrt(cfg["step"]) * np.std(val_res['rec'])))
        # print('val precision: %.4f (%.4f)' % (np.mean(val_res['prec']), 1.96/np.sqrt(cfg["step"]) * np.std(val_res['prec'])))
        
        # with open(os.path.join(save_dir, 'val_metrics.json'), 'w+', encoding='utf8') as f:
        #     f.write(json.dumps(val_res, ensure_ascii=False))
        # print('='*10)
        ### ==============================================================================================================
        print('auc: %.4f (%.4f)' % (np.mean(res['auc']), 1.96/np.sqrt(cfg["step"]) * np.std(res['auc'])))
        print('accuracy: %.4f (%.4f)' % (np.mean(res['acc']), 1.96/np.sqrt(cfg["step"]) * np.std(res['acc'])))
        print('f1 score: %.4f (%.4f)' % (np.mean(res['f1']), 1.96/np.sqrt(cfg["step"]) * np.std(res['f1'])))
        print('recall: %.4f (%.4f)' % (np.mean(res['rec']), 1.96/np.sqrt(cfg["step"]) * np.std(res['rec'])))
        print('precision: %.4f (%.4f)' % (np.mean(res['prec']), 1.96/np.sqrt(cfg["step"]) * np.std(res['prec'])))
        
        with open(os.path.join(save_dir, 'metrics.json'), 'w+', encoding='utf8') as f:
            f.write(json.dumps(res, ensure_ascii=False))

        

def test_model(model, dataloader, loss_fn):
    ## set to eval mode
    model.eval()
    ## test
    for X, y in dataloader:
        pred = model(X)
        val_loss = loss_fn(pred, y)
        loss = val_loss.item()
        
        pred = pred.flatten().cpu().detach().numpy()
        y = y.flatten().cpu().detach().numpy()
        auc = roc_auc_score(y, pred)
        pred = np.where(pred < 0.5, 0, 1)
        f1, rec, prec, acc = f1_score(y, pred), recall_score(y, pred), precision_score(y, pred), accuracy_score(y, pred)
        
        return auc, acc, f1, rec, prec, loss




if __name__ == '__main__':
    args = parse_args()
    train(args)
