import torch
import numpy as np

# Convert pandas dataframe to torch dataset
# Time Point Dataset, only take the last observation of each time series 
class TimePointDataset(torch.utils.data.Dataset):
    def __init__(self, df, feature_cols, label_col):
        
        self.df = df
        self.ids = self.df['id'].unique().tolist()
        
        X_ = []
        y_ = []
        for i in range(len(self.ids)):
            pid = self.ids[i]
            x = self.df[df['id']==pid][feature_cols].to_numpy()
            y = self.df[df['id']==pid][label_col].to_numpy()
            y = np.max(y)
            X_.append(x[-1])
            y_.append(y)
            
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
        feature = self.X[idx]
        label = self.y[idx]
        return feature, label