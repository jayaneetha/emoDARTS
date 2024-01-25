""" Config class for search/augment """
import argparse
import os
from functools import partial

import torch


def get_parser(name):
    """ make default formatted parser """
    parser = argparse.ArgumentParser(name, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # print default value always
    parser.add_argument = partial(parser.add_argument, help=' ')
    return parser


def parse_gpus(gpus):
    if gpus == 'all':
        return list(range(torch.cuda.device_count()))
    else:
        return [int(s) for s in gpus.split(',')]


class BaseConfig(argparse.Namespace):
    def print_params(self, prtf=print):
        prtf("")
        prtf("Parameters:")
        for attr, value in sorted(vars(self).items()):
            prtf("{}={}".format(attr.upper(), value))
        prtf("")

    def as_markdown(self):
        """ Return configs as markdown format """
        text = "|name|value|  \n|-|-|  \n"
        for attr, value in sorted(vars(self).items()):
            text += "|{}|{}|  \n".format(attr, value)

        return text

    def as_dict(self):
        d = {}
        for attr, value in sorted(vars(self).items()):
            d[attr] = value

        return d


class TrainConfig(BaseConfig):
    def build_parser(self):
        parser = get_parser("Train config")
        parser.add_argument('--name', required=True)
        parser.add_argument('--dataset', required=True, help='CIFAR10 / MNIST / FashionMNIST')
        parser.add_argument('--features', required=True, help='mfcc / stft', default='mfcc')
        parser.add_argument('--model', required=True, help='cnn / cnn_lstm / cnn_lstm_att', default='cnn')
        parser.add_argument('--fold', type=int, help='fold in k fold', default=1)
        parser.add_argument('--batch_size', type=int, default=64, help='batch size')
        parser.add_argument('--print_freq', type=int, default=50, help='print frequency')
        parser.add_argument('--gpus', default='0', help='gpu device ids separated by comma. '
                                                        '`all` indicates use all gpus.')
        parser.add_argument('--epochs', type=int, default=50, help='# of training epochs')
        parser.add_argument('--workers', type=int, default=2, help='# of workers')
        parser.add_argument('--seed', type=int, default=2, help='random seed')
        parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')

        return parser

    def __init__(self):
        parser = self.build_parser()
        args = parser.parse_args()
        super().__init__(**vars(args))

        self.data_path = './data/'
        self.path = os.path.join('train', self.name)
        self.plot_path = os.path.join(self.path, 'plots')
        self.gpus = parse_gpus(self.gpus)
