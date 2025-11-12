import torch
import numpy as np

# Convert pandas dataframe to torch dataset
class VTEDataset(torch.utils.data.Dataset):
    def __init__(self, df, feature_cols, time_col, label_col, test=False, infer=False):
        
        self.df = df
        self.infer = infer
        ids_ = self.df['id'].unique().tolist()
        
        X_ = []
        y_ = []
        for i in range(len(ids_)):
            pid = ids_[i]
            
            fs = self.df[df['id']==pid][feature_cols].to_numpy()       
            time = self.df[df['id']==pid][time_col].to_numpy()     
            y = self.df[df['id']==pid][label_col].to_numpy()
            
            y = np.max(y)
                        
            x = []
            isnan = False
            for j in range(len(feature_cols)):
                f = fs[:, j]
                t = time[~np.isnan(f)]
                f = f[~np.isnan(f)]
                if len(f) < 1:
                    isnan = True
                    break
                if np.std(f) == 0 or np.std(t) == 0:
                    corelation = 0
                else:
                    corelation = np.corrcoef(f, t)[0, 1]
                x += [f[0], f[-1], corelation]   
            if isnan:
                continue
                        
            
            X_.append(np.array(x))
            y_.append(y)
            
        if infer or not test:
            self.X = X_
            self.y = y_
            self.ids = ids_
        
        else:
            num_vte = int(np.sum(y_))
            X_v = [] # vte data
            X_nv = [] # non vte data

            for i in range(len(y_)):
                if y_[i]:
                    X_v.append(X_[i])
                else:
                    X_nv.append(X_[i])          
            
            np.random.shuffle(X_nv)
            X_nv = list(X_nv[:num_vte])
            self.X = X_v+X_nv
            self.y = np.hstack([np.ones(num_vte), np.zeros(num_vte)])
            
            zipped = list(zip(self.X, self.y))
            np.random.shuffle(zipped)
            self.X, self.y = zip(*zipped)
            self.X = np.array(self.X)
            self.y = np.array(self.y) 

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if self.infer:
            return self.X[idx], self.y[idx], self.ids[idx]
        return self.X[idx], self.y[idx]