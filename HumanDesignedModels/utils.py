""" Utilities """
import logging
import os
import shutil

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

import preproc
from IEMOCAPDatasetKFold import IemocapDatasetKFold
from MSPIMPROVDatasetKFold import MspImprovDatasetKFold
from MSPPODCASTDatasetKFold import MspPodcastDatasetKFold


# import torchvision.datasets as dset


def get_data(dataset, data_path, cutout_length, validation, features, fold=None):
    """ Get torchvision dataset """
    dataset = dataset.lower()

    if dataset == 'iemocap_kfold':
        dset_cls = IemocapDatasetKFold
        n_classes = 4
    elif dataset == 'mspimprov_kfold':
        dset_cls = MspImprovDatasetKFold
        n_classes = 4
    elif dataset == 'msppodcast_kfold':
        dset_cls = MspPodcastDatasetKFold
        n_classes = 4
    else:
        raise ValueError(dataset)

    trn_transform, val_transform = preproc.data_transforms(dataset, cutout_length)
    trn_data = dset_cls(root=data_path, train=True, download=True, transform=trn_transform, features=features,
                        fold=fold)

    # assuming shape is NHW or NHWC
    shape = trn_data.data.shape
    if dataset == 'iemocap_kfold' or 'mspimprov_kfold' or 'msppodcast_kfold':
        input_channels = 1 if len(shape) == 4 else 1
        assert shape[2] == shape[3], "not expected shape = {}".format(shape)
        input_size = shape[2]
    else:
        input_channels = 3 if len(shape) == 4 else 1
        assert shape[1] == shape[2], "not expected shape = {}".format(shape)
        input_size = shape[1]

    ret = [input_size, input_channels, n_classes, trn_data]
    if validation:  # append validation data
        ret.append(
            dset_cls(root=data_path, train=False, download=True, transform=val_transform, features=features, fold=fold))

    return ret


def get_logger(file_path):
    """ Make python logger """
    # [!] Since tensorboardX use default logger (e.g. logging.info()), we should use custom logger
    logger = logging.getLogger('darts')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger


def param_size(model):
    """ Compute parameter size in MB """
    n_params = sum(
        np.prod(v.size()) for k, v in model.named_parameters() if not k.startswith('aux_head'))
    return n_params / 1024. / 1024.


def num_parameters(model):
    return sum(
        np.prod(v.size()) for k, v in model.named_parameters() if not k.startswith('aux_head'))


class AverageMeter():
    """ Computes and stores the average and current value """

    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update statistics """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def precision(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # one-hot case
    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))

    return res


def accuracy(output, target):
    pred = output.permute((0, 1))
    batch_size = pred.size(0)
    target = target.view((batch_size, -1))

    pred_c = pred.argmax(1)
    target_c = target.argmax(1)

    return pred_c.eq(target_c.expand_as(pred_c)).float().sum(0).mul_(1.0 / batch_size)


def scores(output, target, weights):
    pred = output.argmax(1)
    target = target.argmax(1)
    t = target.cpu()
    p = pred.cpu()
    sample_weights = []
    for i in t:
        sample_weights.append(weights[i.item()])
    wa = accuracy_score(t, p, sample_weight=sample_weights)
    f1 = f1_score(t, p, average="weighted", zero_division=0.0, sample_weight=sample_weights)
    recall_s = recall_score(t, p, average="weighted", zero_division=0.0, sample_weight=sample_weights)
    precision_s = precision_score(t, p, average="weighted", zero_division=0.0, sample_weight=sample_weights)
    return wa, f1, recall_s, precision_s


def save_checkpoint(state, ckpt_dir, is_best=False):
    filename = os.path.join(ckpt_dir, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(ckpt_dir, 'best.pth.tar')
        shutil.copyfile(filename, best_filename)
