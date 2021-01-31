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
    def __init__(self, df, target_col, train=True, testsize=0.2, seed=None):
        """
        df (DataFrame): input dataset
        target_col (str): df column containing the data labels/targets/outputs
        train (bool): give the training set or the test set
        testsize (float): proportion of the data to reserve for the test set
        seed (int): the random seed for the train/test split
        """
        if seed is None:
            seed = torch.Generator().seed()
        self.seed = seed
        self.train = train  # Set to False to get test data
        self.fulldata = df.copy().dropna(how='any')
        self.target_col = target_col
        self.trainidx, self.testidx = self.traintest_split(self.fulldata, testsize, seed)
        self.traindata, self.testdata = (
            self.fulldata.iloc[idx] for idx in (self.trainidx, self.testidx))

    @property
    def data(self):
        if self.train:
            return self.traindata
        else:
            return self.testdata
    
    @property
    def x_data(self):
        return self.data.drop(self.target_col, axis=1)

    @property
    def y_data(self):
        return self.data[self.target_col]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x_item = self.x_data.iloc[index].values
        y_item = self.y_data.iloc[index]

        return x_item, y_item

    def traintest_split(self, data, testsize, seed):
        testlen = int(round(len(data)*testsize))
        trainlen = len(data) - testlen
        trainset, testset = torch.utils.data.random_split(
            data, [trainlen, testlen], generator=torch.Generator().manual_seed(seed))
        return trainset.indices, testset.indices

class Trainer():
    def __init__(self, data, target_col, state=None):
        self.load_state(state)
        seed = self.state.get('seed')
        self.traindata = DfData(data, target_col, train=True, seed=seed)
        self.testdata = DfData(data, target_col, train=False, seed=seed)
        self.init_state()

    @property
    def x_df(self):
        return self.traindata.x_data

    @property
    def y_df(self):
        return self.traindata.y_data

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

    def save_state(self):
        self.state = {
            'epoch': self.epoch,
            'seed': self.traindata.seed,
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
        self.trainloader = DataLoader(self.traindata, batch_size=1, shuffle=True)
        self.testloader = DataLoader(self.testdata, batch_size=1, shuffle=True)
        self.net = self.build_net()
        self.criterion = self.choose_criterion()
        self.optimizer = self.choose_optimizer()
        self.scheduler = self.choose_scheduler()
        self.epoch = self.state.get('epoch', 0)
        self.performance = self.state.get('performance', self.empty_performance)
        self.save_state()

    def train_step(self):
        self.net.train()
        running_loss = 0
        for x_data, target in self.trainloader:
            # Training pass
            self.optimizer.zero_grad()
            output = self.net(x_data[0].float())
            loss = self.criterion(output, target.float())
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
        # Tally metrics
        running_loss /= len(self.trainloader)
        current_lr = self.optimizer.param_groups[0]['lr']
        return running_loss, current_lr

    def test_step(self):
        self.net.eval()
        test_loss = 0
        correct = correct0 = correct1 = 0
        with torch.no_grad():
            for x_data, target in self.testloader:
                # Test pass
                output = self.net(x_data[0].float())
                test_loss += self.criterion(output, target.float()).item()
                if round(output.item()) == target.item():
                    correct += 1
                    if target.item() == 0:
                        correct0 += 1
                    elif target.item() == 1:
                        correct1 += 1

        # Tally metrics
        y_data = self.testdata.y_data
        test_loss /= len(self.testloader)
        accuracy = 100 * correct / len(self.testloader)
        acc_zero = 100 * correct0 / len(y_data[y_data == 0])
        acc_ones = 100 * correct1 / len(y_data[y_data == 1])
        return test_loss, accuracy, acc_zero, acc_ones

    @property
    def empty_performance(self):
        return pd.DataFrame([],
            columns=['lr', 'trainloss', 'testloss', 'accuracy', 'accuracy_0', 'accuracy_1'])

    def train(self, epochs=10):
        ## Train
        for e in range(epochs):
            self.epoch += 1
            trainloss, current_lr = self.train_step()
            testloss, accuracy, acc_zero, acc_ones = self.test_step()

            metrics = {
                'lr': current_lr,
                'trainloss': trainloss,
                'testloss': testloss,
                'accuracy': accuracy,
                'accuracy_0': acc_zero,
                'accuracy_1': acc_ones,
            }
            self.performance = self.performance.append(metrics, ignore_index=True)
            print(f"({self.epoch}) " + "Acc: {accuracy:.1f}%, Acc0: {accuracy_0:.1f}%, ".format(**metrics) +\
                "Acc1: {accuracy_1:.1f}%, Test loss: {testloss:.4f}, ".format(**metrics) +\
                "Train loss: {trainloss:.4f}, LR: {lr:.4f}".format(**metrics))
            self.scheduler.step()
            # self.scheduler.step(running_loss/len(self.trainloader))
        ## Store training state
        self.save_state()

    def predict(self, inputs):
        result = self.net(torch.tensor(inputs.values).float())
        try:
            return result.item()
        except ValueError:
            return result.detach().numpy().T[0]
