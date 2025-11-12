import torch
import os
import json
import matplotlib.pyplot as plt
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from LR import LR
from utils import load_data
import time




def basic(config_path):
    ## load config and data 
    with open(config_path, 'r') as f:
        cfg = json.loads(f.read())
    if isinstance(cfg['data_type'], str):
        data_types = [cfg['data_type']]
    else:
        data_types = cfg['data_type']
    data = load_data(cfg['data_folder'], data_types)

    pids = data['id'].unique()

    vte_pids = []
    nvte_pids = []
    for pid in pids:
        pdata = data[data['id']==pid]
        if np.max(pdata['VTE']) == 1:
            vte_pids.append(pid)
        else:
            nvte_pids.append(pid)
    print('Num patients: ', len(pids))
    print('Num VTE patients: ', len(vte_pids))
    print('Num Non-VTE patients: ', len(nvte_pids))
    
    return data, cfg




def trends(data):
    cols = ['CD4', 'CD8', 'CD4/CD8', 'WBC', 'PLT', 'HB', 'D-dimer', 'FDP']
    # plt.rc('axes', labelsize=18)
    fig, axs = plt.subplots(3, 3, figsize=(15,15))

    for i in range(len(cols)):
        col = cols[i]
        vte_interp = []
        nonvte_interp = []
        for id in data['id'].unique():
            d = data[data['id'] == id]
            y = np.max(d['VTE'].to_numpy())
            d = d[[col, 'date']].dropna()    
            if len(d) < 1:
                continue
            interp = np.interp(np.arange(0, 10, 0.1), d['date'].to_numpy(), d[col].to_numpy())
            
            if y:
                vte_interp.append(interp)
            else:
                nonvte_interp.append(interp)
        
        vte_mean = np.mean(vte_interp, axis=0)    
        vte_std = np.square(np.std(vte_interp, axis=0))
        nonvte_mean = np.mean(nonvte_interp, axis=0)
        nonvte_std = np.square(np.std(nonvte_interp, axis=0))
        
        axs[i//3, i%3].plot(np.arange(0, 10, 0.1), vte_mean, color='deeppink', label='VTE')
        axs[i//3, i%3].plot(np.arange(0, 10, 0.1), nonvte_mean, color='blue', label='Non-VTE')
        # plt.legend()

        axs[i//3, i%3].set_xlabel('t', fontsize=20)
        axs[i//3, i%3].set_xticks([])
        axs[i//3, i%3].set_ylabel(col, fontsize=20)
    plt.tight_layout()
    plt.savefig(f'./results/trends.pdf')
    # plt.savefig(f'./results/trends.png')
    plt.clf()






def loss_curve(checkpoint):
    ## loss curve
    exp_dirs = glob.glob(os.path.join(checkpoint, '*'))

    losses = []
    losses_val = []
    for exp_dir in exp_dirs:
        if os.path.isdir(exp_dir):
            with open(os.path.join(exp_dir, 'metrics.json'), 'r') as f:
                res = json.loads(f.read()) 
            losses.append(res['loss'])
            losses_val.append(res['loss_val'])
    loss = np.mean(losses, axis=0)
    loss_val = np.mean(losses_val, axis=0)
    ## smooth for better visualization =================================
    # loss = np.convolve(loss, np.ones((10,))/10, mode='valid')
    # loss_val = np.convolve(loss_val, np.ones((10,))/10, mode='valid')
    ## =================================================================
    std = np.std(losses, axis=0)[:len(loss)]
    std_val = np.std(losses_val, axis=0)[:len(loss_val)]

    plt.plot(loss, label='train loss')    
    plt.plot(loss_val, label='validation loss')
    plt.fill_between(np.arange(0, len(loss)), loss-std, loss+std, color='lightblue', alpha=0.2)
    plt.fill_between(np.arange(0, len(loss_val)), loss_val-std_val, loss_val+std_val, color='orange', alpha=0.1)

    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('Loss', fontsize=18)
    plt.legend()
    plt.savefig('./results/loss.pdf')    






def ROC(pred_files, legends):

    # pred_files = ['results/DT', 'results/LR', 'results/RF', 'results/SVM']
    # legends = ['Decision Tree', 'Logistic Regression', 'Random Forest', 'SVM']

    for i in range(len(pred_files)):
        file = pred_files[i]
        df = pd.read_csv(os.path.join(file, 'test_res.csv'))
        fpr, tpr, threshold = roc_curve(df['y'].to_list(), df['pred'].to_list())
        
        plt.plot(fpr, tpr, '--', label=legends[i])

    plt.plot([0, 1], [0, 1], '--', color='lightgray')
    plt.legend()
    plt.xlabel('False Positive Rate', fontsize=15)
    plt.ylabel('True Positive Rate', fontsize=15)

    plt.savefig('results/roc.pdf')






def coeffs(checkpoint_dir):
    
    with open(os.path.join(checkpoint_dir, 'config.json'), 'r') as f:
        cfg = json.loads(f.read()) 
    with open(os.path.join(checkpoint_dir, 'metrics.json'), 'r') as f:
        res = json.loads(f.read())
        
    scores = np.mean([res['auc'], res['acc'], res['f1']], axis=0)
    scores = zip(scores, range(len(scores)))
    scores = sorted(scores, key=lambda x: x[0], reverse=True)
    exp = scores[0][1]    
    print('best run at %d' % exp)
        
    save_dir = './results/coeffs'
    os.makedirs(save_dir, exist_ok=True)

    feature_cols = cfg["feature_cols"]
    time_col = cfg["time_col"]
    label_col = cfg["label_col"]
        
    model = LR(3*len(feature_cols))
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, str(exp), 'best.pt'), weights_only=True, map_location='cpu'))

    for name, param in model.named_parameters():
        if '0.weight' in name:
            param = param.cpu().detach().numpy()[0]
            print(param)
            print(feature_cols)






def time_cost(checkpoint_path, config_path, device='cuda:0'):
    # infer time
    with open(config_path, 'r') as f:
        cfg = json.loads(f.read()) 

    feature_cols = cfg["feature_cols"]
    time_col = cfg["time_col"]
    label_col = cfg["label_col"]
        
    model = LR(3*len(feature_cols))
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    model = model.to(device)

    ## init a dummy input
    ts = np.zeros((100, 100))
    for i in range(100):
        for j in range(100):
            X = torch.zeros((1, 24)).to(device)

            t0 = time.time()
            _ = model(X)
            t1 = time.time()
            
            ts[i, j] = t1-t0

    t_mean = np.mean(ts, axis=0)
    t_std = np.std(ts, axis=0)
    with open(os.path.join('./results/infer_time.csv'), 'w+') as f:
        f.writelines(map(lambda x: str(x[0])+','+str(x[1])+'\n', zip(t_mean, t_std)))

