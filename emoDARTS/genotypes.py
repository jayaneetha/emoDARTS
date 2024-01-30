""" Genotypes
    - Genotype: normal/reduce gene + normal/reduce cell output connection (concat)
    - gene: discrete ops information (w/o output connection)
    - dag: real ops (can be mixed or discrete, but Genotype has only discrete information itself)
"""
import os
from collections import namedtuple

import torch
import torch.nn as nn

from models import ops

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat rnn rnn_concat')

PRIMITIVES = [
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',  # identity
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5',
    'none'
]

RNN_PRIMITIVES = ['skip_connect', 'none']

rnn_primitives_cat = os.getenv("RNN_PRIMITIVES", "all").lower()
if rnn_primitives_cat == 'all':
    RNN_PRIMITIVES = [
        'lstm_att_1',
        'lstm_att_2',
        'lstm_1',
        'lstm_2',
        'lstm_3',
        'lstm_4',
        'rnn_att_1',
        'rnn_att_2',
        'rnn_1',
        'rnn_2',
        'rnn_3',
        'rnn_4',
        'skip_connect',
        'none',
    ]
elif rnn_primitives_cat == 'lstm_att':
    RNN_PRIMITIVES = [
        'lstm_att_1',
        'lstm_att_2',
        'skip_connect',
        'none',
    ]
elif rnn_primitives_cat == 'rnn_att':
    RNN_PRIMITIVES = [
        'rnn_att_1',
        'rnn_att_2',
        'skip_connect',
        'none',
    ]
elif rnn_primitives_cat == 'lstm':
    RNN_PRIMITIVES = [
        'lstm_1',
        'lstm_2',
        'lstm_3',
        'lstm_4',
        'skip_connect',
        'none',
    ]
elif rnn_primitives_cat == 'rnn':
    RNN_PRIMITIVES = [
        'rnn_1',
        'rnn_2',
        'rnn_3',
        'rnn_4',
        'skip_connect',
        'none',
    ]

print("RNN_PRIMITIVES", RNN_PRIMITIVES)


def to_dag(C_in, gene, reduction):
    """ generate discrete ops from gene """
    dag = nn.ModuleList()
    for edges in gene:
        row = nn.ModuleList()
        for op_name, s_idx in edges:
            # reduction cell & from input nodes => stride = 2
            stride = 2 if reduction and s_idx < 2 else 1
            op = ops.OPS[op_name](C_in, stride, True)
            if not isinstance(op, ops.Identity):  # Identity does not use drop path
                op = nn.Sequential(
                    op,
                    ops.DropPath_(module_type='cnn')
                )
            op.s_idx = s_idx
            row.append(op)
        dag.append(row)

    return dag


def to_rnn_dag(C_in, gene):
    dag = nn.ModuleList()
    for edges in gene:
        row = nn.ModuleList()
        for op_name, s_idx in edges:
            op = ops.RNN_OPS[op_name](C_in)
            if not isinstance(op, ops.Identity):  # Identity does not use drop path
                op = nn.Sequential(
                    op,
                    ops.Extract_RNN_Tensor(),
                    # ops.DropPath_(module_type='rnn')
                )
            op.s_idx = s_idx
            row.append(op)
        dag.append(row)

    return dag


def from_str(s):
    """ generate genotype from string
    e.g. "Genotype(
            normal=[[('sep_conv_3x3', 0), ('sep_conv_3x3', 1)],
                    [('sep_conv_3x3', 1), ('dil_conv_3x3', 2)],
                    [('sep_conv_3x3', 1), ('sep_conv_3x3', 2)],
                    [('sep_conv_3x3', 1), ('dil_conv_3x3', 4)]],
            normal_concat=range(2, 6),
            reduce=[[('max_pool_3x3', 0), ('max_pool_3x3', 1)],
                    [('max_pool_3x3', 0), ('skip_connect', 2)],
                    [('max_pool_3x3', 0), ('skip_connect', 2)],
                    [('max_pool_3x3', 0), ('skip_connect', 2)]],
            reduce_concat=range(2, 6))"
    """

    genotype = eval(s)

    return genotype


def parse(alpha, k):
    """
    parse continuous alpha to discrete gene.
    alpha is ParameterList:
    ParameterList [
        Parameter(n_edges1, n_ops),
        Parameter(n_edges2, n_ops),
        ...
    ]

    gene is list:
    [
        [('node1_ops_1', node_idx), ..., ('node1_ops_k', node_idx)],
        [('node2_ops_1', node_idx), ..., ('node2_ops_k', node_idx)],
        ...
    ]
    each node has two edges (k=2) in CNN.
    """

    gene = []
    assert PRIMITIVES[-1] == 'none'  # assume last PRIMITIVE is 'none'

    # 1) Convert the mixed op to discrete edge (single op) by choosing top-1 weight edge
    # 2) Choose top-k edges per node by edge score (top-1 weight in edge)
    for edges in alpha:
        # edges: Tensor(n_edges, n_ops)
        edge_max, primitive_indices = torch.topk(edges[:, :-1], 1)  # ignore 'none'
        topk_edge_values, topk_edge_indices = torch.topk(edge_max.view(-1), k)
        node_gene = []
        for edge_idx in topk_edge_indices:
            prim_idx = primitive_indices[edge_idx]
            prim = PRIMITIVES[prim_idx]
            node_gene.append((prim, edge_idx.item()))

        gene.append(node_gene)

    return gene


def parse_rnn(alpha, k):
    gene = []
    assert RNN_PRIMITIVES[-1] == 'none'  # assume last PRIMITIVE is 'none'

    # 1) Convert the mixed op to discrete edge (single op) by choosing top-1 weight edge
    # 2) Choose top-k edges per node by edge score (top-1 weight in edge)
    for edges in alpha:
        # edges: Tensor(n_edges, n_ops)
        edge_max, primitive_indices = torch.topk(edges[:, :-1], 1)  # ignore 'none'
        topk_edge_values, topk_edge_indices = torch.topk(edge_max.view(-1), k)
        node_gene = []
        for edge_idx in topk_edge_indices:
            prim_idx = primitive_indices[edge_idx]
            prim = RNN_PRIMITIVES[prim_idx]
            node_gene.append((prim, edge_idx.item()))

        gene.append(node_gene)

    return gene
