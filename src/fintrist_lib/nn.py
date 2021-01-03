"""Construct Neural Networks using Pytorch"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn

from .base import RecipeBase

class NNModel():
    def __init__(self):
        pass

    def build_net(self):
        net = Net()
        print(net)
        return net
        
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


class TrendData(Dataset):
    def __init__(self, df):
        df = df.copy().dropna(how='any')
        df['up tomorrow'] = (df['days to gain'] == 1).astype(int)
        self.df = df
        self.target = 'up tomorrow'

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        x_df = self.df.drop(['days to gain', self.target], axis=1)
        y_df = self.df[self.target]
        x_data = x_df.iloc[index].values
        y_data = y_df.iloc[index]

        return x_data, y_data

dataset = TrendData(inputdata)
traindata = DataLoader(dataset, batch_size=1, shuffle=True)
traindata

## Training
# criterion = nn.L1Loss()
# criterion = nn.MSELoss()
criterion = nn.SmoothL1Loss()
# Optimizers require the parameters to optimize and a learning rate
optimizer = torch.optim.SGD(net.parameters(), lr=0.03)
epochs = 100
for e in range(epochs):
    running_loss = 0
    for x_data, labels in traindata:
        # Training pass
        optimizer.zero_grad()
        
        output = net(x_data[0].float())
        loss = criterion(output, labels.float())
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss/len(traindata)}")


## Inspect results
result = net(torch.tensor(dataset.df.drop(['days to gain', dataset.target], axis=1).values).float())
labels = dataset.df['days to gain'].values

result_np = result.detach().numpy()
df = inputdata.dropna(how='any')
df['up prediction'] = result_np.T[0]
