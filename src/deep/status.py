'''utils for training status'''

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import logging
import time, datetime
import subprocess
import pprint
import warnings

import matplotlib.pyplot as plt
import torch
from torch.utils.collect_env import get_pretty_env_info
from tqdm import tqdm

from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

'''Value Collector'''

class Meter(list):
    '''collect values'''
    def __init__(self, name: str, *args):
        super().__init__(args)
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def x(self, total: int=None):
        '''return x axis for plot

        Arguments:
            total: int (default: None)
                total length of x axis
                if given, x will be evenly distributed.
        '''
        if total is None:
            return range(1, self.__len__()+1)
        per_element = total // self.__len__()
        return range(per_element, total+1, per_element)

class Group(dict):
    def max_length(self):
        return max([len(v) for v in self.values()])

class Collector:
    '''Collect scalar values and plot them

    Structure:
        {
            'group1': {
                'key1' : [...],
                'key2' : [...]},
            ...
        }
    same group => will be plotted in same graph.

    Usage:
        key1 = 'Loss/train/g'
        key2 = 'Loss/train/d'
        #      |----------|-|
        #         group   | key

        collector = Collector()
        # initialize collector
        collector.initialize(key1, key2)
        # add values
        collector['Loss/train/g'].append(random.random())
        collector['Loss/train/d'].append(random.random())
        # plot
        collector.plot()
        # => image of 1x<number of groups> graph
    '''
    def __init__(self) -> None:
        self._groups = {}
        self._initialized = False

    @property
    def initialized(self):
        return self._initialized

    def _split_key(self, key: str) -> tuple[str, str]:
        key = key.split('/')
        return '/'.join(key[:-1]), key[-1]

    def initialize(self, *keys) -> None:
        for key in keys:
            self[key] = Meter(key)
        self._initialized = True

    def update_by_dict(self, step: dict):
        for key, value in step.items():
            self[key].append(value)

    def plot(self, filename: str='graph.jpg') -> None:
        col = self.__len__()

        fig, axes = plt.subplots(1, col,
            figsize=(7*col, 5), tight_layout=True)

        for i, group_name in enumerate(self):
            if col == 1: ax = axes
            else:        ax = axes[i]

            group = self[group_name]
            length = group.max_length()
            legends = []
            for key in group:
                legends.append(key)
                x, y = group[key].x(length), group[key]
                ax.plot(x, y)

            ax.set_title(group_name)
            ax.legend(legends, loc='upper right')
            ax.set_xlabel('iterations')

        plt.savefig(filename)
        plt.close()

    '''magic funcs'''

    def __getitem__(self, key: str) -> Any:
        if key in self._groups:
            return self._groups[key]
        group, key = self._split_key(key)
        return self._groups[group][key]

    def __setitem__(self,
        key: str, value: Any
    ) -> None:
        group, key = self._split_key(key)
        if group not in self._groups:
            self._groups[group] = Group()
        self._groups[group][key] = value

    def __iter__(self) -> Iterable:
        return self._groups.__iter__()

    def __len__(self) -> int:
        return self._groups.__len__()

    def __str__(self) -> str:
        return self._groups.__str__()

'''Training Status'''

class Status:
    '''Status
    A class for keeping training status

    Arguments:
        max_iters: int
            maximum iteration to train
        bar: bool (default: True)
            if True, show bar by tqdm
        log_file: str (default: None)
            path to the log file
            if given, log status to a file
        log_interval: int (default: 1)
            interval for writing to log file
        logger_name: str (default: 'logger')
            name for logger
    '''
    def __init__(self,
        max_iters: int, bar: bool=True,
        log_file: str=None, log_interval: int=1, logger_name: str='logger'
    ) -> None:
        if bar:
            self.bar = tqdm(total=max_iters)
        self._max_iters = max_iters
        self._batches_done = 0
        self._collector = Collector()
        self._log_file = log_file
        if log_file is not None:
            logging.basicConfig(
                filename=log_file, filemode='w',
                format='%(asctime)s:%(filename)s:%(levelname)s: %(message)s')
            self._logger = logging.getLogger(logger_name)
            self._logger.setLevel(logging.DEBUG)
        self._log_interval = log_interval
        self._step_start = time.time()

    @property
    def max_iters(self):
        return self._max_iters
    @property
    def batches_done(self):
        return self._batches_done
    @batches_done.setter
    def batches_done(self, value):
        self._batches_done = value

    def print(self, *args, **kwargs):
        '''print function'''
        if hasattr(self, 'bar'):
            tqdm.write(*args, **kwargs)
        else:
            print(*args, **kwargs)
    def log(self, message, level='info'):
        if hasattr(self, '_logger'):
            getattr(self._logger, level)(message)
        else:
            warnings.warn('No Logger. Printing to stdout.')
            self.print(message)

    '''Information loggers'''

    def log_args(self, args):
        dict_args = pprint.pformat(vars(args))
        self.log(f'Command line arguments\n{dict_args}')

    def log_torch(self):
        env = get_pretty_env_info()
        self.log(f'PyTorch environment:\n{env}')

    def log_models(self, *models):
        for model in models:
            self.log(f'Architecture: {model.__class__.__name__}\n{model}')

    def log_gpu(self):
        if torch.cuda.is_available():
            nvidia_smi_output = subprocess.run(
                'nvidia-smi', shell=True,
                capture_output=True, universal_newlines=True)
            self.log(f'\n{nvidia_smi_output.stdout}')
        else:
            self.log('No GPU available on your enviornment.')

    def log_training(self, args, *models):
        '''log information in one function'''
        if args is not None:
            self.log_args(args)
        self.log_torch()
        self.log_models(*models)

    '''a step'''

    def update(self, **kwargs) -> None:
        '''update status'''
        if not self._collector.initialized:
            self._collector.initialize(*list(kwargs.keys()))

        self.update_collector(**kwargs)
        postfix = [f'{k} : {v:.5f}' for k, v in kwargs.items()]

        # log
        if self._log_file is not None \
            and self.batches_done % self._log_interval == 0:
            # FIXME: this ETA is not exact.
            duration = time.time() - self._step_start
            eta_sec = int((self.max_iters - self.batches_done) * duration)
            eta = datetime.timedelta(seconds=eta_sec)
            self.log(
                f'STEP: {self.batches_done} / {self.max_iters} INFO: {kwargs} ETA: {eta}')
        if self.batches_done == 0:
            # print gpu on first step
            # for checking memory usage
            self.log_gpu()

        self.batches_done += 1
        self._step_start = time.time()

        if hasattr(self, 'bar'):
            self.bar.set_postfix_str(' '.join(postfix))
            self.bar.update(1)

    def initialize_collector(self, *keys):
        if not self._collector.initialized:
            self._collector.initialize(*keys)
    def update_collector(self, **kwargs):
        self._collector.update_by_dict(kwargs)

    def is_end(self):
        '''have reached last batch?'''
        return self.batches_done >= self.max_iters

    def load_state_dict(self, state_dict: dict) -> None:
        '''fast forward'''
        # load
        self._collector = state_dict['collector']
        self.batches_done = state_dict['batches_done']
        if self.batches_done > 0:
            # fastforward progress bar if present
            if hasattr(self, 'bar'):
                self.bar.update(self.batches_done)

    def state_dict(self) -> dict:
        return dict(
            collector=self._collector,
            batches_done=self.batches_done)

    def plot_loss(self, filename='loss'):
        self._collector.plot(filename)

def log_test(
    print_fn,
    targets,
    predictions,
    labels: list=None,
    filename: str=None
):
    if labels is not None:
        labels = np.array(labels)
        targets = labels[targets]
        predictions = labels[predictions]

    print_fn('TEST')
    # precision, recall, F1, accuracy
    print_fn(
        f'Classification report:\n{classification_report(targets, predictions)}')

    # confusion matrix
    confmat = confusion_matrix(targets, predictions, labels=labels, normalize='true')
    print_fn(f'Confusion matrix:\n{confmat}')

    if filename is not None:
        # visualize
        fig, ax = plt.subplots(tight_layout=True)
        ax.matshow(confmat, cmap='Blues')
        for (i, j), value in np.ndenumerate(confmat):
            ax.text(j, i, f'{value:.3f}', ha='center', va='center')
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlabel('prediction')
        ax.set_ylabel('ground truth')
        plt.savefig(filename)
        plt.close()
        print_fn(f'Saved confusion matrix to {filename}')
