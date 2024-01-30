""" Operations """
import genotypes as gt
import torch
import torch.nn as nn
import torch.nn.functional as F

OPS = {
    'none': lambda C, stride, affine: Zero(stride),
    'avg_pool_3x3': lambda C, stride, affine: PoolBN('avg', C, 3, stride, 1, affine=affine),
    'max_pool_3x3': lambda C, stride, affine: PoolBN('max', C, 3, stride, 1, affine=affine),
    'skip_connect': lambda C, stride, affine: \
        Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    'sep_conv_3x3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
    'sep_conv_5x5': lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
    'sep_conv_7x7': lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
    'dil_conv_3x3': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),  # 5x5
    'dil_conv_5x5': lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),  # 9x9
    'conv_7x1_1x7': lambda C, stride, affine: FacConv(C, C, 7, stride, 3, affine=affine)
}

RNN_OPS = {
    'lstm_1': lambda C: LSTMOp(C, 1),
    'lstm_2': lambda C: LSTMOp(C, 2),
    'lstm_3': lambda C: LSTMOp(C, 3),
    'lstm_4': lambda C: LSTMOp(C, 4),
    'skip_connect': lambda C: Identity(),
    'rnn_1': lambda C: RNNOp(C, 1),
    'rnn_2': lambda C: RNNOp(C, 2),
    'rnn_3': lambda C: RNNOp(C, 3),
    'rnn_4': lambda C: RNNOp(C, 4),
    'lstm_att_1': lambda C: LSTMAttentionOp(C, num_layers=1),
    'lstm_att_2': lambda C: LSTMAttentionOp(C, num_layers=2),
    'rnn_att_1': lambda C: RNNAttentionOp(C, num_layers=1),
    'rnn_att_2': lambda C: RNNAttentionOp(C, num_layers=2),
    'none': lambda C: Zero(1),
}


class Extract_RNN_Tensor(nn.Module):
    def forward(self, x):
        # Output shape (batch, features, hidden)
        try:
            tensor, _ = x
        except ValueError:
            tensor = x
        # Reshape shape (batch, hidden)
        return tensor


def drop_path_(x, drop_prob, training, module_type='cnn'):
    if training and drop_prob > 0.:
        keep_prob = 1. - drop_prob
        # per data point mask; assuming x in cuda.
        if module_type == 'cnn':
            mask = torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
            x.div_(keep_prob).mul_(mask)
        elif module_type == 'rnn':
            mask = torch.cuda.FloatTensor(x.size(0), 1).bernoulli_(keep_prob)
            x.div_(keep_prob).mul_(mask)

    return x


class DropPath_(nn.Module):
    def __init__(self, module_type='cnn', p=0.):
        """ [!] DropPath is inplace module
        Args:
            p: probability of an path to be zeroed.
        """
        super().__init__()
        self.p = p
        self.module_type = module_type

    def extra_repr(self):
        return 'p={}, inplace'.format(self.p)

    def forward(self, x):
        drop_path_(x, self.p, self.training, self.module_type)

        return x


class PoolBN(nn.Module):
    """
    AvgPool or MaxPool - BN
    """

    def __init__(self, pool_type, C, kernel_size, stride, padding, affine=True):
        """
        Args:
            pool_type: 'max' or 'avg'
        """
        super().__init__()
        if pool_type.lower() == 'max':
            self.pool = nn.MaxPool2d(kernel_size, stride, padding)
        elif pool_type.lower() == 'avg':
            self.pool = nn.AvgPool2d(kernel_size, stride, padding, count_include_pad=False)
        else:
            raise ValueError()

        self.bn = nn.BatchNorm2d(C, affine=affine)

    def forward(self, x):
        # print(f"PoolBN input: {x.shape}")
        out = self.pool(x)
        out = self.bn(out)
        # print(f"PoolBN output: {out.shape}")
        return out


class StdConv(nn.Module):
    """ Standard conv
    ReLU - Conv - BN
    """

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(C_in, C_out, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        # print(f"StdConv input: {x.shape}")
        out = self.net(x)
        # print(f"StdConv output: {out.shape}")

        return out


class FacConv(nn.Module):
    """ Factorized conv
    ReLU - Conv(Kx1) - Conv(1xK) - BN
    """

    def __init__(self, C_in, C_out, kernel_length, stride, padding, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(C_in, C_in, (kernel_length, 1), stride, padding, bias=False),
            nn.Conv2d(C_in, C_out, (1, kernel_length), stride, padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        # print(f"FacConv input: {x.shape}")
        out = self.net(x)
        # print(f"FacConv output: {out.shape}")
        return out


class DilConv(nn.Module):
    """ (Dilated) depthwise separable conv
    ReLU - (Dilated) depthwise separable - Pointwise - BN

    If dilation == 2, 3x3 conv => 5x5 receptive field
                      5x5 conv => 9x9 receptive field
    """

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(C_in, C_in, kernel_size, stride, padding, dilation=dilation, groups=C_in,
                      bias=False),
            nn.Conv2d(C_in, C_out, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        # print(f"DilConv input: {x.shape}")
        out = self.net(x)
        # print(f"DilConv output: {out.shape}")
        return out


class SepConv(nn.Module):
    """ Depthwise separable conv
    DilConv(dilation=1) * 2
    """

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            DilConv(C_in, C_in, kernel_size, stride, padding, dilation=1, affine=affine),
            DilConv(C_in, C_out, kernel_size, 1, padding, dilation=1, affine=affine)
        )

    def forward(self, x):
        # # print(f"SepConv Input: {x.shape}")
        # print(f"SepConv input: {x.shape}")
        out = self.net(x)
        # print(f"SepConv output: {out.shape}")
        return out


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x * 0.

        # re-sizing by stride
        return x[:, :, ::self.stride, ::self.stride] * 0.


class FactorizedReduce(nn.Module):
    """
    Reduce feature map size by factorized pointwise(stride=2).
    """

    def __init__(self, C_in, C_out, affine=True):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        # print(f"FactorizedReduce input: {x.shape}")
        x = self.relu(x)
        out = torch.cat([self.conv1(x), self.conv2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        # print(f"FactorizedReduce output: {out.shape}")
        return out


class LSTMOp(nn.Module):
    def __init__(self, C, num_lstm_layers):
        super().__init__()
        self.lstm = nn.LSTM(C, C, num_layers=num_lstm_layers, batch_first=True)

    def forward(self, x):
        x = self.lstm(x)
        return x


class RNNOp(nn.Module):
    def __init__(self, C, num_lstm_layers):
        super().__init__()
        self.rnn = nn.RNN(C, C, num_layers=num_lstm_layers, batch_first=True)

    def forward(self, x):
        x = self.rnn(x)
        return x


class LSTMAttentionOp(nn.Module):
    def __init__(self, C, num_layers):
        super().__init__()
        self.hidden_unit = C
        # ATTENTION
        att_layer_input_size = (2 * self.hidden_unit) + (self.hidden_unit // 4 * num_layers)
        self.attn = nn.Linear(att_layer_input_size, self.hidden_unit * 2)
        self.attn_combine = nn.Linear(self.hidden_unit * 4, self.hidden_unit * 2)

        # LAYER 1 - input layer
        self.lstm_layer = nn.LSTM(input_size=C,
                                  hidden_size=self.hidden_unit,
                                  num_layers=num_layers,
                                  dropout=0.3,
                                  batch_first=True,
                                  bidirectional=True)

        self.out = nn.Linear(self.hidden_unit * 2, self.hidden_unit)

    def forward(self, x):
        batch_size = x.size(0)
        output, (h_n, c_n) = self.lstm_layer(x)

        last_output = torch.cat((h_n, c_n), 1)
        last_output = last_output.view(batch_size, -1)

        output_main = torch.cat((last_output, output), 1)

        # ------------------ATTENTION-----------------------------
        attn_conc = self.attn(output_main)
        attn_weights = F.softmax(attn_conc, dim=1)
        attn_applied = torch.mul(attn_weights, output)
        output = torch.cat((output, attn_applied), 1)
        output = self.attn_combine(output)
        return self.out(output)


class RNNAttentionOp(nn.Module):
    def __init__(self, C, num_layers):
        super().__init__()
        self.hidden_unit = C
        # ATTENTION
        att_layer_input_size = (2 * self.hidden_unit) + (self.hidden_unit // 8 * num_layers)
        self.attn = nn.Linear(att_layer_input_size, self.hidden_unit * 2)
        self.attn_combine = nn.Linear(self.hidden_unit * 4, self.hidden_unit * 2)

        # LAYER 1 - input layer
        self.rnn_layer = nn.RNN(input_size=C,
                                hidden_size=self.hidden_unit,
                                num_layers=num_layers,
                                dropout=0.3,
                                batch_first=True,
                                bidirectional=True)

        self.out = nn.Linear(self.hidden_unit * 2, self.hidden_unit)

    def forward(self, x):
        batch_size = x.size(0)
        output, h_n = self.rnn_layer(x)

        last_output = h_n
        last_output = last_output.view(batch_size, -1)

        output_main = torch.cat((last_output, output), 1)

        # ------------------ATTENTION-----------------------------
        attn_conc = self.attn(output_main)
        attn_weights = F.softmax(attn_conc, dim=1)
        attn_applied = torch.mul(attn_weights, output)
        output = torch.cat((output, attn_applied), 1)
        output = self.attn_combine(output)
        return self.out(output)


class MixedOp(nn.Module):
    """ Mixed operation """

    def __init__(self, C, stride):
        super().__init__()
        self._ops = nn.ModuleList()
        for primitive in gt.PRIMITIVES:
            op = OPS[primitive](C, stride, affine=False)
            self._ops.append(op)

    def forward(self, x, weights):
        """
        Args:
            x: input
            weights: weight for each operation
        """
        output = sum(w * op(x) for w, op in zip(weights, self._ops))
        return output


class RNNMixedOp(nn.Module):
    def __init__(self, C):
        super().__init__()
        self._ops = nn.ModuleList()
        for primitive in gt.RNN_PRIMITIVES:
            op = RNN_OPS[primitive](C)
            self._ops.append(op)

    def forward(self, x, weights):
        outputs = []
        for w, op in zip(weights, self._ops):
            op_output = op(x)
            try:
                o, _ = op_output
            except ValueError:
                o = op_output
            op_w = w * o
            outputs.append(op_w)

        output = sum(outputs)
        return output
