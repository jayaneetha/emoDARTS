""" CNN for architecture search """
import logging

import genotypes as gt
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.search_cells import SearchCell, RNNSearchCell
from torch.nn.parallel._functions import Broadcast


def broadcast_list(l, device_ids):
    """ Broadcasting list """
    l_copies = Broadcast.apply(device_ids, *l)
    l_copies = [l_copies[i:i + len(l)] for i in range(0, len(l_copies), len(l))]

    return l_copies


class SearchCNN(nn.Module):
    """ Search CNN model """

    def __init__(self, C_in, C, n_classes, n_layers, n_nodes=4, stem_multiplier=3, n_nodes_rnn=4, n_layers_rnn=4):
        """
        Args:
            C_in: # of input channels
            C: # of starting model channels
            n_classes: # of classes
            n_layers: # of layers
            n_nodes: # of intermediate nodes in Cell
            stem_multiplier
        """
        super().__init__()

        self.C_in = C_in
        self.C = C
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.n_layers_rnn = n_layers_rnn

        C_cur = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(C_in, C_cur, 3, 1, 1, bias=False),
            nn.BatchNorm2d(C_cur)
        )

        # for the first cell, stem is used for both s0 and s1
        # [!] C_pp and C_p is output channel size, but C_cur is input channel size.
        C_pp, C_p, C_cur = C_cur, C_cur, C

        self.cells = nn.ModuleList()
        reduction_p = False
        for i in range(n_layers):
            # Reduce featuremap size and double channels in 1/3 and 2/3 layer.
            if i in [n_layers // 3, 2 * n_layers // 3]:
                C_cur *= 2
                reduction = True
            else:
                reduction = False

            cell = SearchCell(n_nodes, C_pp, C_p, C_cur, reduction_p, reduction)
            reduction_p = reduction
            self.cells.append(cell)
            C_cur_out = C_cur * n_nodes
            C_pp, C_p = C_p, C_cur_out

        self.gap_0 = nn.AdaptiveAvgPool2d(1)
        self.gap_1 = nn.AdaptiveAvgPool2d(1)

        self.linear = nn.Linear(C_p, n_classes)

        self.rnn_cells = nn.ModuleList()

        for i in range(n_layers_rnn):
            cell = RNNSearchCell(n_nodes_rnn, C_p)
            self.rnn_cells.append(cell)

    def forward(self, x, weights_normal, weights_reduce, weights_rnn):
        s0 = s1 = self.stem(x)

        for cell in self.cells:
            weights = weights_reduce if cell.reduction else weights_normal
            s0, s1 = s1, cell(s0, s1, weights)

        rnn_s1 = self.gap_1(s1)
        rnn_s1 = rnn_s1.view(rnn_s1.size(0), -1)  # flatten

        rnn_s0 = self.gap_0(s0)
        rnn_s0 = rnn_s0.view(rnn_s0.size(0), -1)

        for cell in self.rnn_cells:
            rnn_s0, rnn_s1 = rnn_s1, cell(rnn_s0, rnn_s1, weights_rnn)

        logits = self.linear(rnn_s1)

        return logits


class SearchCNNController(nn.Module):
    """ SearchCNN controller supporting multi-gpu """

    def __init__(self, C_in, C, n_classes, n_layers, criterion, n_nodes=4, stem_multiplier=3,
                 device_ids=None, n_layers_rnn=4, n_nodes_rnn=4):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_nodes_rnn = n_nodes_rnn
        self.criterion = criterion
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        self.device_ids = device_ids

        # initialize architect parameters: alphas
        n_ops = len(gt.PRIMITIVES)
        n_ops_rnn = len(gt.RNN_PRIMITIVES)

        self.alpha_normal = nn.ParameterList()
        self.alpha_reduce = nn.ParameterList()
        self.alpha_rnn = nn.ParameterList()

        for i in range(n_nodes):
            self.alpha_normal.append(nn.Parameter(1e-3 * torch.randn(i + 2, n_ops)))
            self.alpha_reduce.append(nn.Parameter(1e-3 * torch.randn(i + 2, n_ops)))

        for i in range(n_nodes_rnn):
            self.alpha_rnn.append(nn.Parameter(1e-3 * torch.randn(i + 2, n_ops_rnn)))

        # setup alphas list
        self._alphas = []
        for n, p in self.named_parameters():
            if 'alpha' in n:
                self._alphas.append((n, p))

        self.net = SearchCNN(C_in, C, n_classes, n_layers, n_nodes, stem_multiplier, n_layers_rnn=n_layers_rnn,
                             n_nodes_rnn=n_nodes_rnn)

    def forward(self, x):
        weights_normal = [F.softmax(alpha, dim=-1) for alpha in self.alpha_normal]
        weights_reduce = [F.softmax(alpha, dim=-1) for alpha in self.alpha_reduce]
        weights_rnn = [F.softmax(alpha, dim=-1) for alpha in self.alpha_rnn]

        if len(self.device_ids) == 1:
            return self.net(x, weights_normal, weights_reduce, weights_rnn)

        # scatter x
        xs = nn.parallel.scatter(x, self.device_ids)
        # broadcast weights
        wnormal_copies = broadcast_list(weights_normal, self.device_ids)
        wreduce_copies = broadcast_list(weights_reduce, self.device_ids)

        # replicate modules
        replicas = nn.parallel.replicate(self.net, self.device_ids)
        outputs = nn.parallel.parallel_apply(replicas,
                                             list(zip(xs, wnormal_copies, wreduce_copies)),
                                             devices=self.device_ids)
        return nn.parallel.gather(outputs, self.device_ids[0])

    def loss(self, X, y):
        logits = self.forward(X)
        return self.criterion(logits, y)

    def print_alphas(self, logger):
        # remove formats
        org_formatters = []
        for handler in logger.handlers:
            org_formatters.append(handler.formatter)
            handler.setFormatter(logging.Formatter("%(message)s"))

        logger.info("####### ALPHA #######")
        logger.info("# Alpha - normal")
        for alpha in self.alpha_normal:
            logger.info(F.softmax(alpha, dim=-1))

        logger.info("\n# Alpha - reduce")
        for alpha in self.alpha_reduce:
            logger.info(F.softmax(alpha, dim=-1))
        logger.info("#####################")

        logger.info("\n# Alpha - Rnn")
        for alpha in self.alpha_rnn:
            logger.info(F.softmax(alpha, dim=-1))
        logger.info("#####################")

        # restore formats
        for handler, formatter in zip(logger.handlers, org_formatters):
            handler.setFormatter(formatter)

    def genotype(self):
        gene_normal = gt.parse(self.alpha_normal, k=2)
        gene_reduce = gt.parse(self.alpha_reduce, k=2)
        gene_rnn = gt.parse_rnn(self.alpha_rnn, k=2)
        concat = range(2, 2 + self.n_nodes)  # concat all intermediate nodes
        concat_rnn = range(2, 2 + self.n_nodes_rnn)

        return gt.Genotype(normal=gene_normal, normal_concat=concat,
                           reduce=gene_reduce, reduce_concat=concat,
                           rnn=gene_rnn, rnn_concat=concat_rnn)

    def weights(self):
        return self.net.parameters()

    def named_weights(self):
        return self.net.named_parameters()

    def alphas(self):
        for n, p in self._alphas:
            yield p

    def named_alphas(self):
        for n, p in self._alphas:
            yield n, p
