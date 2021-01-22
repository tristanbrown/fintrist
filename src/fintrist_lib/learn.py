"""Construct Neural Networks using Pytorch"""
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn

from .base import RecipeBase

__all__ = ['Net', 'DfData', 'Trainer']
        
class Net(nn.Module):
    def __init__(self, inputs, depth=4, width=None, outputs=1, output_type='bounded'):
        super().__init__()
        if width is None:
            width = inputs * 2
        if output_type == 'bounded':
            self.final_act = torch.sigmoid
        else:
            self.final_act = torch.relu

        self.layers = self.build_layers(inputs, depth, width, outputs)
        self.architecture = {
            'inputs': inputs, 'depth': depth, 'width': width, 'outputs': outputs,
            'output_type': output_type}

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
    def __init__(self, data, target_col, state=None):
        self.data_df = DfData(data, target_col)
        self.load_state(state)
        self.init_state()

    @property
    def x_df(self):
        return self.data_df.x_df

    @property
    def y_df(self):
        return self.data_df.y_df

    def apply_state_dict(self, obj, state_name):
        if old_state := self.state.get(state_name):
            obj.load_state_dict(old_state)

    def build_net(self, **kwargs):
        if kwargs.get('inputs') is None:
            kwargs['inputs'] = len(self.x_df.columns)
        net_architecture = self.state.get('architecture', {})
        net_architecture.update(kwargs)
        net = Net(**net_architecture)
        self.apply_state_dict(net, 'model')
        print(net)
        return net

    def switch_net(self, depth, width, outputs, output_type):
        self.net = self.build_net(depth, width, outputs, output_type)
        self.save_state()
        self.init_state()

    def choose_criterion(self):
        criterion = nn.SmoothL1Loss()
        self.apply_state_dict(criterion, 'criterion')
        return criterion

    def choose_optimizer(self):
        optimizer = torch.optim.SGD(self.net.parameters(), lr=0.01)
        self.apply_state_dict(optimizer, 'optimizer')
        return optimizer

    def choose_scheduler(self):
        sched_type = self.state.get('scheduler_type', 'CyclicLR')
        if sched_type == 'CyclicLR':
            defaults = {
                'base_lr': 0.0001,
                'max_lr': 0.01,
                'step_size_up': 5,
                'mode': 'exp_range',
                'gamma': 0.99,
            }
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                self.optimizer, **defaults)
        elif sched_type == 'ReduceLROnPlateau':
            defaults = {
                'mode': 'max',
                'factor': 0.5,
                'patience': 5,
                'verbose': True,
            }
        SchedulerCls = getattr(torch.optim.lr_scheduler, sched_type)
        scheduler = SchedulerCls(self.optimizer, **defaults)
        self.apply_state_dict(scheduler, 'scheduler')
        return scheduler

    @property
    def empty_performance(self):
        return pd.DataFrame([], columns=['epoch', 'loss', 'lr'])

    def save_state(self):
        self.state = {
            'epoch': self.epoch,
            'architecture': self.net.architecture,
            'model': self.net.state_dict(),
            'criterion': self.criterion.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'scheduler_type': self.scheduler.__class__.__name__,
            'performance': self.performance
        }

    def load_state(self, state):
        if state == None:
            state = {}
        self.state = state

    def init_state(self):
        self.traindata = DataLoader(self.data_df, batch_size=1, shuffle=True)
        self.net = self.build_net()
        self.criterion = self.choose_criterion()
        self.optimizer = self.choose_optimizer()
        self.scheduler = self.choose_scheduler()
        self.epoch = self.state.get('epoch', 0)
        self.performance = self.state.get('performance', self.empty_performance)
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

            loss_metric = running_loss/len(self.traindata)
            current_lr = self.optimizer.param_groups[0]['lr']
            metrics = {
                'epoch': self.epoch,
                'loss': loss_metric,
                'lr': current_lr,
            }
            self.performance.append(metrics, ignore_index=True)
            print("Epoch: {epoch}, Training loss: {loss}, LR: {lr}".format(**metrics))
            self.scheduler.step()
            # self.scheduler.step(running_loss/len(self.traindata))
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
