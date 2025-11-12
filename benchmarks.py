import os
import torch
import json
import time
import shutil
import numpy as np
import pandas as pd
from utils import load_data, cancer_types, init_resdict, norm_df_ignorena, split_data
from datasets.TimePoint_dataset import TimePointDataset
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import argparse




def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--config', type=str, default='./configs/config_benchmarks.json', help='path to benchmark config file')
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--infer', action='store_true', default=False, help='if save results')

    args = parser.parse_args()
    return args



def test_LR(train_X, train_y, test_X, test_y, save_dir=None, max_iter=1000):
    model = LogisticRegression(max_iter=max_iter, solver='liblinear')
    model.fit(train_X, train_y)

    pred = model.predict(test_X)
    f1, rec, prec, acc = f1_score(test_y, pred), recall_score(test_y, pred), precision_score(test_y, pred), accuracy_score(test_y, pred)
    pred = model.predict_proba(test_X)[:, 1:]
    auc = roc_auc_score(test_y, pred)
    
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        
        pred = model.predict_proba(train_X)[:, 1:]
        df = pd.DataFrame(np.hstack([train_y.reshape((-1, 1)), pred]), columns=['y', 'pred'])
        df.to_csv(os.path.join(save_dir, 'train_res.csv'), index=False)
        
        pred = model.predict_proba(test_X)[:, 1:]
        df = pd.DataFrame(np.hstack([test_y.reshape((-1, 1)), pred]), columns=['y', 'pred'])
        df.to_csv(os.path.join(save_dir, 'test_res.csv'), index=False)
    
    return auc, f1, rec, prec, acc


def test_SVM(train_X, train_y, test_X, test_y, save_dir=None):
    model = SVC(gamma='auto', probability=True)
    model.fit(train_X, train_y)

    pred = model.predict(test_X)
    f1 = f1_score(test_y, pred)
    rec = recall_score(test_y, pred)
    prec = precision_score(test_y, pred)
    acc = accuracy_score(test_y, pred)
    
    pred = model.predict_proba(test_X)[:, 1:]
    auc = roc_auc_score(test_y, pred)
    
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        
        pred = model.predict(train_X)
        df = pd.DataFrame(np.hstack([train_y.reshape((-1, 1)), pred.reshape((-1, 1))]), columns=['y', 'pred'])
        df.to_csv(os.path.join(save_dir, 'train_res.csv'), index=False)
        
        pred = model.predict(test_X)
        df = pd.DataFrame(np.hstack([test_y.reshape((-1, 1)), pred.reshape((-1, 1))]), columns=['y', 'pred'])
        df.to_csv(os.path.join(save_dir, 'test_res.csv'), index=False)
        
    return auc, f1, rec, prec, acc


def test_DecisionTree(train_X, train_y, test_X, test_y, save_dir=None):
    model = DecisionTreeClassifier()
    model.fit(train_X, train_y)

    pred = model.predict(test_X)
    f1, rec, prec, acc = f1_score(test_y, pred), recall_score(test_y, pred), precision_score(test_y, pred), accuracy_score(test_y, pred)
    pred = model.predict_proba(test_X)[:, 1:]
    auc = roc_auc_score(test_y, pred)
    
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        
        pred = model.predict_proba(train_X)[:, 1:]
        df = pd.DataFrame(np.hstack([train_y.reshape((-1, 1)), pred]), columns=['y', 'pred'])
        df.to_csv(os.path.join(save_dir, 'train_res.csv'), index=False)
        
        pred = model.predict_proba(test_X)[:, 1:]
        df = pd.DataFrame(np.hstack([test_y.reshape((-1, 1)), pred]), columns=['y', 'pred'])
        df.to_csv(os.path.join(save_dir, 'test_res.csv'), index=False)
    
    return auc, f1, rec, prec, acc


def test_RF(train_X, train_y, test_X, test_y, save_dir=None):
    model = RandomForestClassifier(max_depth=3)
    model.fit(train_X, train_y)

    pred = model.predict(test_X)
    f1, rec, prec, acc = f1_score(test_y, pred), recall_score(test_y, pred), precision_score(test_y, pred), accuracy_score(test_y, pred)
    pred = model.predict_proba(test_X)[:, 1:]
    auc = roc_auc_score(test_y, pred)
    
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        
        pred = model.predict_proba(train_X)[:, 1:]
        df = pd.DataFrame(np.hstack([train_y.reshape((-1, 1)), pred]), columns=['y', 'pred'])
        df.to_csv(os.path.join(save_dir, 'train_res.csv'), index=False)
        
        pred = model.predict_proba(test_X)[:, 1:]
        df = pd.DataFrame(np.hstack([test_y.reshape((-1, 1)), pred]), columns=['y', 'pred'])
        df.to_csv(os.path.join(save_dir, 'test_res.csv'), index=False)
    
    return auc, f1, rec, prec, acc


def train_benchmarks(args):
    
    ## load config and data
    with open(args.config, 'r') as f:
        cfg = json.loads(f.read()) 
    if isinstance(cfg['data_type'], str):
        data_types = [cfg['data_type']]
    else:
        data_types = cfg['data_type']
    feature_cols = cfg["feature_cols"] + cancer_types
    label_col = cfg["label_col"]
    
    data = load_data(cfg['data_folder'], data_types)
    
    ## normalization
    if 'means' in cfg and 'stds' in cfg:
        data, means, stds = norm_df_ignorena(data, feature_cols, cfg['means'], cfg['stds'])
    else:
        data, means, stds = norm_df_ignorena(data, feature_cols)
        cfg['means'] = means.tolist()
        cfg['stds'] = stds.tolist()
    ## fill missing data with 0
    data.fillna(0, inplace=True)
    
    ## init res dict
    lr_res = init_resdict()
    svm_res = init_resdict()
    dt_res = init_resdict()
    rf_res = init_resdict()
    
    ## run test
    for i in range(cfg['step']):
        
        # split dataset for each run
        train_data, _, test_data = split_data(data)
        
        # build dataset
        train_dataset = TimePointDataset(train_data, feature_cols, label_col)
        test_dataset = TimePointDataset(test_data, feature_cols, label_col)
        
        auc, f1, rec, prec, acc = test_LR(train_dataset.X, train_dataset.y, test_dataset.X, test_dataset.y)
        lr_res['auc'].append(auc)
        lr_res['f1'].append(f1)
        lr_res['rec'].append(rec)
        lr_res['prec'].append(prec)
        lr_res['acc'].append(acc)
        
        auc, f1, rec, prec, acc = test_SVM(train_dataset.X, train_dataset.y, test_dataset.X, test_dataset.y)
        svm_res['auc'].append(auc)
        svm_res['f1'].append(f1)
        svm_res['rec'].append(rec)
        svm_res['prec'].append(prec)
        svm_res['acc'].append(acc)
        
        auc, f1, rec, prec, acc = test_DecisionTree(train_dataset.X, train_dataset.y, test_dataset.X, test_dataset.y)
        dt_res['auc'].append(auc)
        dt_res['f1'].append(f1)
        dt_res['rec'].append(rec)
        dt_res['prec'].append(prec)
        dt_res['acc'].append(acc)
        
        auc, f1, rec, prec, acc = test_RF(train_dataset.X, train_dataset.y, test_dataset.X, test_dataset.y)
        rf_res['auc'].append(auc)
        rf_res['f1'].append(f1)
        rf_res['rec'].append(rec)
        rf_res['prec'].append(prec)
        rf_res['acc'].append(acc)
        
        
    print('Logistic Regression: ')
    print('auc: %.4f (%.4f)' % (np.mean(lr_res['auc']), 1.96/np.sqrt(cfg["step"]) * np.std(lr_res['auc'])))
    print('f1 score: %.4f (%.4f)' % (np.mean(lr_res['f1']), 1.96/np.sqrt(cfg["step"]) * np.std(lr_res['f1'])))
    print('recall: %.4f (%.4f)' % (np.mean(lr_res['rec']), 1.96/np.sqrt(cfg["step"]) * np.std(lr_res['rec'])))
    print('precision: %.4f (%.4f)' % (np.mean(lr_res['prec']), 1.96/np.sqrt(cfg["step"]) * np.std(lr_res['prec'])))
    print('accuracy: %.4f (%.4f)' % (np.mean(lr_res['acc']), 1.96/np.sqrt(cfg["step"]) * np.std(lr_res['acc'])))

    print('SVM: ')
    print('auc: %.4f (%.4f)' % (np.mean(svm_res['auc']), 1.96/np.sqrt(cfg["step"]) * np.std(svm_res['auc'])))
    print('f1 score: %.4f (%.4f)' % (np.mean(svm_res['f1']), 1.96/np.sqrt(cfg["step"]) * np.std(svm_res['f1'])))
    print('recall: %.4f (%.4f)' % (np.mean(svm_res['rec']), 1.96/np.sqrt(cfg["step"]) * np.std(svm_res['rec'])))
    print('precision: %.4f (%.4f)' % (np.mean(svm_res['prec']), 1.96/np.sqrt(cfg["step"]) * np.std(svm_res['prec'])))
    print('accuracy: %.4f (%.4f)' % (np.mean(svm_res['acc']), 1.96/np.sqrt(cfg["step"]) * np.std(svm_res['acc'])))
    
    print('Decision Tree: ')
    print('auc: %.4f (%.4f)' % (np.mean(dt_res['auc']), 1.96/np.sqrt(cfg["step"]) * np.std(dt_res['auc'])))
    print('f1 score: %.4f (%.4f)' % (np.mean(dt_res['f1']), 1.96/np.sqrt(cfg["step"]) * np.std(dt_res['f1'])))
    print('recall: %.4f (%.4f)' % (np.mean(dt_res['rec']), 1.96/np.sqrt(cfg["step"]) * np.std(dt_res['rec'])))
    print('precision: %.4f (%.4f)' % (np.mean(dt_res['prec']), 1.96/np.sqrt(cfg["step"]) * np.std(dt_res['prec'])))
    print('accuracy: %.4f (%.4f)' % (np.mean(dt_res['acc']), 1.96/np.sqrt(cfg["step"]) * np.std(dt_res['acc'])))

    print('Random Forest: ')
    print('auc: %.4f (%.4f)' % (np.mean(rf_res['auc']), 1.96/np.sqrt(cfg["step"]) * np.std(rf_res['auc'])))
    print('f1 score: %.4f (%.4f)' % (np.mean(rf_res['f1']), 1.96/np.sqrt(cfg["step"]) * np.std(rf_res['f1'])))
    print('recall: %.4f (%.4f)' % (np.mean(rf_res['rec']), 1.96/np.sqrt(cfg["step"]) * np.std(rf_res['rec'])))
    print('precision: %.4f (%.4f)' % (np.mean(rf_res['prec']), 1.96/np.sqrt(cfg["step"]) * np.std(rf_res['prec'])))
    print('accuracy: %.4f (%.4f)' % (np.mean(rf_res['acc']), 1.96/np.sqrt(cfg["step"]) * np.std(rf_res['acc'])))
    
    if args.infer:
        test_LR(train_dataset.X, train_dataset.y, test_dataset.X, test_dataset.y, save_dir=os.path.join(args.save_dir, 'LR'))
        test_DecisionTree(train_dataset.X, train_dataset.y, test_dataset.X, test_dataset.y, save_dir=os.path.join(args.save_dir, 'DT'))
        test_RF(train_dataset.X, train_dataset.y, test_dataset.X, test_dataset.y, save_dir=os.path.join(args.save_dir, 'RF'))
        test_SVM(train_dataset.X, train_dataset.y, test_dataset.X, test_dataset.y, save_dir=os.path.join(args.save_dir, 'SVM'))
    
    
    
if __name__ == '__main__':
    args = parse_args()
    train_benchmarks(args)