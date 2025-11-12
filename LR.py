from torch import nn

## Logistic Regression Model Implemented Using Pytorch
class LR(nn.Module):
    def __init__(self, feature_in, hidden_dims=[]):
        super(LR, self).__init__()
        
        ## build hidden layers
        hidden_dims = [feature_in] + hidden_dims
        layers = []
        for i in range(1, len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)
        
        ## output layer
        self.out_layer = nn.Sequential(
            nn.Linear(hidden_dims[-1], 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.out_layer(self.layers(x))