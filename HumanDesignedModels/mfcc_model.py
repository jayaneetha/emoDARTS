import torch
import torch.nn.functional as F
from torch import nn


class MFCC_CNNModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cnn = nn.Conv2d(1, 1, 2, 2, 2)
        self.mp = nn.MaxPool2d(2, 2)
        self.dr = nn.Dropout(0.3)
        self.linear1 = nn.Linear(33 * 33 * 1, 32)
        self.linear2 = nn.Linear(32, 4)

    def forward(self, input_logits):
        cnn = self.cnn(input_logits)
        mp = self.mp(cnn)
        fl = torch.flatten(mp, start_dim=1)
        l1 = self.linear1(fl)
        dr = self.dr(l1)
        l2 = self.linear2(dr)
        output = nn.functional.softmax(l2, dim=0)
        return output


class MFCC_RNNModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.recurrent_layer = nn.RNN(128 * 128, 512, 3)
        self.linear1 = nn.Linear(512, 32)
        self.linear2 = nn.Linear(32, 4)

    def forward(self, input_logits):
        batch_size = input_logits.shape[0]
        input_logits = input_logits.reshape(batch_size, 1, -1)

        rnn_lo, _ = self.recurrent_layer(input_logits)
        fl2 = torch.flatten(rnn_lo, start_dim=1)
        l1 = self.linear1(fl2)
        l2 = self.linear2(l1)
        output = nn.functional.softmax(l2, dim=0)
        return output


class MFCC_LSTMModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lstm_layer = nn.LSTM(128 * 128, 512, 1)
        self.linear1 = nn.Linear(512, 32)
        self.linear2 = nn.Linear(32, 4)

    def forward(self, input_logits):
        batch_size = input_logits.shape[0]
        input_logits = input_logits.reshape(batch_size, 1, -1)

        lstm, _ = self.lstm_layer(input_logits)
        fl2 = torch.flatten(lstm, start_dim=1)
        l1 = self.linear1(fl2)
        l2 = self.linear2(l1)
        output = nn.functional.softmax(l2, dim=0)
        return output


class MFCC_CNNLSTMModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cnn = nn.Conv2d(1, 1, 2, 2, 2)
        self.mp = nn.MaxPool2d(2, 2)
        self.dr = nn.Dropout(0.3)
        self.lstm_layer = nn.LSTM(33 * 33 * 1, 128, 1, bidirectional=True)
        self.linear1 = nn.Linear(128 * 2, 32)
        self.linear2 = nn.Linear(32, 4)

    def forward(self, input_logits):
        cnn = self.cnn(input_logits)
        mp = self.mp(cnn)
        fl = torch.flatten(mp, start_dim=1)
        lstm, _ = self.lstm_layer(fl)
        l1 = self.linear1(lstm)
        dr = self.dr(l1)
        l2 = self.linear2(dr)
        output = nn.functional.softmax(l2, dim=0)
        return output


class MFCC_CNNLSTMAttModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden_unit = 128
        num_layers = 1
        self.cnn = nn.Conv2d(1, 1, 2, 2, 2)
        self.mp = nn.MaxPool2d(2, 2)
        self.dr = nn.Dropout(0.3)
        self.lstm_layer = nn.LSTM(33 * 33 * 1, 128, num_layers, batch_first=True, bidirectional=True)

        # ATTENTION
        att_layer_input_size = (2 * self.hidden_unit) + (self.hidden_unit // 4 * num_layers)
        self.attn = nn.Linear(att_layer_input_size, self.hidden_unit * 2)
        self.attn_combine = nn.Linear(self.hidden_unit * 4, self.hidden_unit * 2)

        self.out = nn.Linear(self.hidden_unit * 2, 4)

    def forward(self, input_logits):
        batch_size = input_logits.size(0)

        cnn = self.cnn(input_logits)
        mp = self.mp(cnn)
        fl = torch.flatten(mp, start_dim=1)
        output, (h_n, c_n) = self.lstm_layer(fl)
        last_output = torch.cat((h_n, c_n), 1)
        last_output = last_output.view(batch_size, -1)
        output_main = torch.cat((last_output, output), 1)

        # ------------------ATTENTION-----------------------------
        attn_conc = self.attn(output_main)
        attn_weights = F.softmax(attn_conc, dim=1)
        attn_applied = torch.mul(attn_weights, output)
        output = torch.cat((output, attn_applied), 1)
        output = self.attn_combine(output)
        out = self.out(output)
        out = nn.functional.softmax(out, dim=0)
        return out
