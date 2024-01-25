from functools import reduce

import torch

from constant import DEVICE

device = torch.device(DEVICE)


def main():
    model = 'cnnlstm'
    features = 'mfcc'
    # set default gpu device id
    torch.cuda.set_device(0)

    torch.backends.cudnn.benchmark = True

    model = get_model(model, features).to(device)

    parameter_count = pytorch_count_params(model)
    print(parameter_count)


def pytorch_count_params(model):
    "count number trainable parameters in a pytorch model"
    total_params = sum(reduce(lambda a, b: a * b, x.size()) for x in model.parameters())
    return total_params


def get_model(model_type, features):
    model_type = model_type.lower()
    features = features.lower()

    if features == 'mfcc':
        from mfcc_model import MFCC_CNNModel, MFCC_RNNModel, MFCC_LSTMModel, MFCC_CNNLSTMModel, MFCC_CNNLSTMAttModel
        if model_type == 'cnn':
            return MFCC_CNNModel()
        elif model_type == 'rnn':
            return MFCC_RNNModel()
        elif model_type == 'lstm':
            return MFCC_LSTMModel()
        elif model_type == 'cnnlstm':
            return MFCC_CNNLSTMModel()
        elif model_type == 'cnnlstmatt':
            return MFCC_CNNLSTMAttModel()
        else:
            raise Exception('invalid model type')
    else:
        raise Exception('invalid feature')


if __name__ == "__main__":
    main()
