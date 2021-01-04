"""Construct Neural Networks using Pytorch"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn

from .base import RecipeBase

__all__ = ['Net', 'DfData']
        
class Net(nn.Module):
    def __init__(self, inputs, depth, width=None, outputs=1, bounded=True):
        super().__init__()
        if width is None:
            width = inputs * 2
        if bounded:
            self.final_act = torch.sigmoid
        else:
            self.final_act = torch.relu

        self.layers = self.build_layers(inputs, depth, width, outputs)

    def build_layers(self, inputs, depth, width, outputs):
        layers = []
        conns = [inputs] + [width - i*(width-outputs)//(depth-1) for i in range(depth - 1)] + [outputs]
        for i in range(depth):
            layers.append(nn.Linear(conns[i], conns[i+1]))
            layers.append(nn.ReLU())
        layers = layers[:-1]
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # forward pass
        x = self.final_act(self.layers(x))
        return x


class DfData(Dataset):
    def __init__(self, df, target_col):
        self.df = df.copy().dropna(how='any')
        self.x_df = self.df.drop(target_col, axis=1)
        self.y_df = self.df[target_col]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        x_data = self.x_df.iloc[index].values
        y_data = self.y_df.iloc[index]

        return x_data, y_data

## Inspect results
# result = net(torch.tensor(dataset.df.drop(['days to gain', dataset.target], axis=1).values).float())
# labels = dataset.df['days to gain'].values

# result_np = result.detach().numpy()
# df = inputdata.dropna(how='any')
# df['up prediction'] = result_np.T[0]
