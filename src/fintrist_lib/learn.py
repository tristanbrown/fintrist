"""Construct Neural Networks using Pytorch"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn

from .base import RecipeBase

__all__ = ['Net', 'DfData', 'Trainer']
        
class Net(nn.Module):
    def __init__(self, inputs, depth=4, width=None, outputs=1, bounded=True):
        super().__init__()
        if width is None:
            width = inputs * 2
        if bounded:
            self.final_act = torch.sigmoid
        else:
            self.final_act = torch.relu

        self.layers = self.build_layers(inputs, depth, width, outputs)
        self.architecture = {
            'inputs': inputs, 'depth': depth, 'width': width, 'outputs': outputs,
            'bounded': bounded}

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

class Trainer():
    def __init__(self, data, target_col, net=None, **kwargs):
        self.data_df = DfData(data, target_col)
        self._net = net or self.build_net()
        self.state = {}
        self.init_training(**kwargs)

    @property
    def x_df(self):
        return self.data_df.x_df

    @property
    def y_df(self):
        return self.data_df.y_df

    def build_net(self, **kwargs):
        if kwargs.get('inputs') is None:
            kwargs['inputs'] = len(self.x_df.columns)
        net = Net(**kwargs)
        print(net)
        return net

    def init_training(self, lr=0.1):
        self.traindata = DataLoader(self.data_df, batch_size=1, shuffle=True)
        self.criterion = nn.SmoothL1Loss()
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=epochs // 5, verbose=True)
        self.epoch = 0

    def save_state(self):
        self.state = {
            'epoch': self.epoch,
            'architecture': self.net.architecture,
            'model': self.net.state_dict(),
            'criterion': self.criterion.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }

    def load_state(self, state):
        self.net = self.build_net(**state.get('architecture'))
        self.net.load_state_dict(state.get('model'))
        self.criterion.load_state_dict(state.get('criterion'))
        self.optimizer.load_state_dict(state.get('optimizer'))
        self.scheduler.load_state_dict(state.get('scheduler'))
        self.epoch = state.get('epoch', 0)
        self.save_state()

    def train(self, epochs=10):
        ## Train
        for e in range(epochs):
            running_loss = 0
            self.epoch += 1
            for x_data, labels in self.traindata:
                # Training pass
                self.optimizer.zero_grad()

                output = self.net(x_data[0].float())
                loss = self.criterion(output, labels.float())
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
            else:
                print(f"Training loss: {running_loss/len(self.traindata)}")
            self.scheduler.step(running_loss/len(self.traindata))
        ## Store training state
        self.save_state()

    def predict(self, inputs):
        pass

## Inspect results
# result = net(torch.tensor(dataset.df.drop(['days to gain', dataset.target], axis=1).values).float())
# labels = dataset.df['days to gain'].values

# result_np = result.detach().numpy()
# df = inputdata.dropna(how='any')
# df['up prediction'] = result_np.T[0]
