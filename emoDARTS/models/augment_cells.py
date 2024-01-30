""" CNN cell for network augmentation """
import genotypes as gt
import torch
import torch.nn as nn
from models import ops


class AugmentCell(nn.Module):
    """ Cell for augmentation
    Each edge is discrete.
    """

    def __init__(self, genotype, C_pp, C_p, C, reduction_p, reduction):
        super().__init__()
        self.reduction = reduction
        self.n_nodes = len(genotype.normal)

        if reduction_p:
            self.preproc0 = ops.FactorizedReduce(C_pp, C)
        else:
            self.preproc0 = ops.StdConv(C_pp, C, 1, 1, 0)
        self.preproc1 = ops.StdConv(C_p, C, 1, 1, 0)

        # generate dag
        if reduction:
            gene = genotype.reduce
            self.concat = genotype.reduce_concat
        else:
            gene = genotype.normal
            self.concat = genotype.normal_concat

        self.dag = gt.to_dag(C, gene, reduction)

        self.dropout = nn.Dropout(0.3)

    def forward(self, s0, s1):
        s0 = self.preproc0(s0)
        s1 = self.preproc1(s1)

        states = [s0, s1]
        for edges in self.dag:
            s_cur = sum(op(states[op.s_idx]) for op in edges)
            states.append(s_cur)

        s_out = torch.cat([states[i] for i in self.concat], dim=1)

        # Add a dropout after each cell.
        # was: return s_out
        return self.dropout(s_out)


class RNNAugmentCell(nn.Module):
    def __init__(self, genotype, C):
        super().__init__()
        self.rnn_n_nodes = len(genotype.rnn)

        gene = genotype.rnn
        self.concat = genotype.rnn_concat

        self.dag = gt.to_rnn_dag(C, gene)
        self.gap = nn.AdaptiveAvgPool1d(128)
        self.dropout = nn.Dropout(0.3)

    def forward(self, s0, s1):
        states = [s0, s1]
        for edges in self.dag:
            outputs = []
            for op in edges:
                x = states[op.s_idx]
                o = op(x)
                outputs.append(o)

            s_cur = sum(outputs)

            states.append(s_cur)

        s_out = torch.cat([states[i] for i in self.concat], dim=1)
        s_out = self.gap(s_out)
        return self.dropout(s_out)
