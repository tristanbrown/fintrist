"""Construct Neural Networks using Pytorch"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, WeightedRandomSampler
from torch import nn
from copy import deepcopy

from .base import RecipeBase

__all__ = ['Net', 'DfData', 'Trainer']

class Net(nn.Module):
    def __init__(self, inputs, depth=4, width=None, outputs=1, output_type='bounded',
                activation='relu', normalize=None):
        super().__init__()
        if width is None:
            width = inputs * 2

        self.normalize = normalize

        # Define the activation functions between layers.
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU()
        elif activation == 'selu':
            self.activation = nn.SELU()

        # Define the final activation function.
        if output_type == 'bounded':
            self.final_act = nn.Sigmoid()
        elif output_type == 'logit':
            self.final_act = None
        else:
            self.final_act = self.activation

        # Build the layers
        self.layers = self.build_layers(inputs, depth, width, outputs)
        self.architecture = {
            'inputs': inputs, 'depth': depth, 'width': width, 'outputs': outputs,
            'output_type': output_type}

    def build_layers(self, inputs, depth, width, outputs):
        layers = []
        conns = [inputs] + [width - i*(width-outputs)//(depth-1) for i in range(depth - 1)] + [outputs]
        if (self.normalize == 'inputs') or (self.normalize == 'layer'):
            layers.append(nn.LayerNorm(conns[0]))
        for i in range(depth):
            layers.append(nn.Linear(conns[i], conns[i+1]))
            layers.append(self.activation)
            if (self.normalize == 'layer') and (i < list(range(depth))[-1]):
                layers.append(nn.LayerNorm(conns[i+1]))
        layers = layers[:-1]
        if self.final_act:
            layers.append(self.final_act)
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # forward pass
        x = self.layers(x)
        return x


class DfData(Dataset):
    def __init__(self, dataset, target_col, test=False, testsize=0.2, seed=None):
        """
        dataset (DataFrame): input dataset
        target_col (str): df column containing the data labels/targets/outputs
        train (bool): give the training set or the test set
        testsize (float): proportion of the data to reserve for the test set
        seed (int): the random seed for the train/test split
        """
        if seed is None:
            seed = torch.Generator().seed()
        self.seed = seed
        self.fulldata = dataset.copy().dropna(how='any')
        self.target_col = target_col
        self.trainidx, self.testidx = self.traintest_split(self.fulldata, testsize, seed)
        self.traindata, self.testdata = (
            self.fulldata.iloc[idx] for idx in (self.trainidx, self.testidx))
        self.set_test(test)

    def set_test(self, test=True):
        self.test = test
        self.reload_data()

    def reload_data(self):
        self.x_data = torch.tensor(self.x_df.values)
        self.y_data = torch.tensor(self.y_df)

    @property
    def data(self):
        if self.test:
            return self.testdata
        else:
            return self.traindata

    @property
    def x_df(self):
        return self.data.drop(self.target_col, axis=1)

    @property
    def y_df(self):
        return self.data[self.target_col]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x_item = self.x_data[index]
        y_item = self.y_data[index]

        return x_item, y_item

    def traintest_split(self, data, testsize, seed):
        testlen = int(round(len(data)*testsize))
        trainlen = len(data) - testlen
        trainset, testset = torch.utils.data.random_split(
            data, [trainlen, testlen], generator=torch.Generator().manual_seed(seed))
        return trainset.indices, testset.indices


class Trainer():
    def __init__(self, data, target_col, **state):
        self.output_type = None
        self.load_state(state)
        seed = self.state.get('seed')
        self.traindata = DfData(data, target_col, test=False, seed=seed)
        self.testdata = deepcopy(self.traindata)
        self.testdata.set_test(True)

    ## State init and update

    def load_state(self, state):
        if state == None:
            state = {}
        self.state = state

    def save_state(self):
        self.state = {
            'epoch': self.epoch,
            'seed': self.traindata.seed,
            'batch_size': self.batch_size,
            'balance': self.balance,
            'architecture': self.net.architecture,
            'model': self.net.state_dict(),
            'criterion': self.criterion.state_dict(),
            'crit_type': self.criterion.__class__.__name__,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'scheduler_type': self.scheduler.__class__.__name__,
            'performance': self.performance
        }

    def init_state(self, stateargs, resume=False):
        if batch_size := stateargs.get('batch_size'):
            self.state['batch_size'] = batch_size
        if (balance := stateargs.get('balance')) is not None:
            self.state['balance'] = balance
        if nn_params := stateargs.get('architecture'):
            self.state['architecture'] = nn_params
        self.batch_size = self.state.get('batch_size', 1)
        self.balance = self.state.get('balance', False)
        self.trainloader = DataLoader(
            self.traindata, batch_size=self.batch_size,
            sampler=self.choose_sampler(self.traindata))
        self.testloader = DataLoader(self.testdata, batch_size=1)
        self.net = self.build_net()
        self.criterion = self.update_criterion(stateargs.get('criterion', {}))
        self.optimizer = self.choose_optimizer()
        self.scheduler = self.choose_scheduler(stateargs.get('scheduler', {}))
        self.epoch = self.state.get('epoch', 0)
        self.performance = self.state.get('performance', self.empty_performance)
        self.save_state()

    def apply_state_dict(self, obj, state_name):
        """Take the saved state and apply it to the object."""
        try:
            if old_state := self.state.get(state_name):
                obj.load_state_dict(old_state)
        except (RuntimeError, ValueError):
            pass

    def update_state_dict(self, obj, state_name, params):
        """Update parameters in the state and apply them to the object."""
        old_state = self.state.get(state_name)
        old_state.update(params)
        self.apply_state_dict(obj, state_name)

    ## Choosing Configurables

    def choose_sampler(self, dataset):
        """Choose whether to rebalance the classes in the dataset.

        Balancing occurs through replacement sampling.
        """
        if self.balance:
            y = dataset.y_df
            counts = np.bincount(y)
            weights = 1 / torch.Tensor(counts)
            class_weights = weights[y]
            sampler = WeightedRandomSampler(
                class_weights, len(class_weights), replacement=True)
        else:
            sampler = RandomSampler(dataset)
        return sampler

    def build_net(self, **kwargs):
        if kwargs.get('inputs') is None:
            kwargs['inputs'] = len(self.traindata.x_df.columns)
        if output_type := getattr(self, 'output_type', None):
            kwargs['output_type'] = output_type
        net_architecture = self.state.get('architecture', {})
        net_architecture.update(kwargs)
        net = Net(**net_architecture)
        self.apply_state_dict(net, 'model')
        return net

    def update_criterion(self, params):
        label = params.get('label') or params.get('type')
        weight = params.get('weight')
        return self.choose_criterion(crit_type=label, weight=weight)

    def choose_criterion(self, crit_type=None, weight=None):
        if crit_type is None:
             crit_type = self.state.get('crit_type', 'SmoothL1Loss')
        if crit_type.lower() in ['smoothl1loss', 'regression', 'l1']:
            criterion = nn.SmoothL1Loss()
        elif crit_type.lower() in ['bcewithlogitsloss', 'binary', 'bce']:
            target_counts = self.traindata.y_df.value_counts()
            if weight is None:
                weight = torch.tensor([target_counts[0]/target_counts[1]])
            criterion = nn.BCEWithLogitsLoss(weight)
            self.output_type = 'logit'
        elif crit_type.lower() in ['crossentropyloss', 'crossentropy', 'classifcation']:
            # Need to implement a calculation for weight here.
            criterion = nn.CrossEntropyLoss()
            self.output_type = 'softmax' # Need to implement this on the Net.
        self.apply_state_dict(criterion, 'criterion')
        return criterion

    def choose_optimizer(self):
        optimizer = torch.optim.SGD(self.net.parameters(), lr=0.01)
        self.apply_state_dict(optimizer, 'optimizer')
        return optimizer

    def choose_scheduler(self, params):
        sched_type = params.pop('type', self.state.get('scheduler_type', 'CyclicLR'))
        statedict = self.state.get('scheduler', {})
        if sched_type == 'CyclicLR':
            defaults = {
                'base_lr': 0.0001,
                'max_lr': 0.01,
                'step_size_up': 5,
                'mode': 'exp_range',
                'gamma': 0.99,
            }
            # Necessary because load_state_dict updates gamma without changing scaling
            if statedict:
                new_dict = {
                    'base_lr': statedict['base_lrs'][0],
                    'max_lr': statedict['max_lrs'][0],
                    'step_size_up': statedict['step_ratio'] * statedict['total_size'],
                    'mode': statedict['mode'],
                    'gamma': statedict['gamma'],
                }
                defaults.update(new_dict)
        elif sched_type == 'ReduceLROnPlateau':
            defaults = {
                'mode': 'max',
                'factor': 0.5,
                'patience': 5,
                'verbose': True,
            }
        defaults.update(self.sched_param_map(params))
        SchedulerCls = getattr(torch.optim.lr_scheduler, sched_type)
        scheduler = SchedulerCls(self.optimizer, **defaults)

        # Necessary to resume state without overwriting gamma scaling.
        if statedict:
            statedict.pop('scale_fn', None)
            self.apply_state_dict(scheduler, 'scheduler')
        return scheduler

    def sched_param_map(self, params):
        mappings = {
            'lr': 'max_lr',
            'min_lr': 'base_lr',
            'step_size': 'step_size_up',
        }
        for k, v in mappings.items():
            value = params.pop(k, None)
            if value:
                params[v] = value
        return params

    ## Training the NN

    def train_step(self):
        self.net.train()
        running_loss = 0
        for x_data, target in self.trainloader:
            # Training pass
            self.optimizer.zero_grad()
            output = self.net(x_data.float()).squeeze(1)
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
                output = self.net(x_data.float()).squeeze(1)
                test_loss += self.criterion(output, target.float()).item()
                if self.output_type == 'logit':
                    output = torch.sigmoid(output)
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
        if self.epoch == 0:
            # Drop the 1st step to align base_lr with save points.
            self.scheduler.step()
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
                "Train loss: {trainloss:.4f}, LR: {lr:.6f}".format(**metrics))
            self.scheduler.step()
        ## Store training state
        self.save_state()

    def lr_rangetest(self):
        min_lr = 0.0001
        max_lr = 0.1
        epochs = 200
        gamma = (max_lr/min_lr)**(1/epochs)
        self.optimizer = self.choose_optimizer()
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=1,
            gamma=gamma
        )
        self.train(epochs)

    ## Inference

    def predict(self, inputs, shuffle_col=None):
        if shuffle_col:
            inputs = inputs.copy()
            inputs[shuffle_col] = inputs[shuffle_col].sample(frac=1).values
        result = self.net(torch.tensor(inputs.values).float())
        if self.output_type == 'logit':
            result = torch.sigmoid(result)
        try:
            return result.item()
        except ValueError:
            return result.detach().numpy().T[0]
