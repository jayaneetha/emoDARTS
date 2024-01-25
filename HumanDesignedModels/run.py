import os

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import utils
from config import TrainConfig

config = TrainConfig()

device = torch.device("cuda")

# tensorboard
writer = SummaryWriter(log_dir=os.path.join(config.path, "tb"))
writer.add_text('config', config.as_markdown(), 0)

logger = utils.get_logger(os.path.join(config.path, "{}.log".format(config.name)))
config.print_params(logger.info)


def main():
    logger.info("Logger is set - training start")

    # set default gpu device id
    torch.cuda.set_device(config.gpus[0])

    # set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    torch.backends.cudnn.benchmark = True

    input_size, input_channels, n_classes, train_data, valid_data = utils.get_data(
        config.dataset, config.data_path, cutout_length=0, validation=True, features=config.features, fold=config.fold)

    model = get_model(config.model, config.features).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    n_train = len(train_data)
    split = n_train // 5
    indices = list(range(n_train))

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=config.batch_size,
                                               shuffle=False,
                                               num_workers=config.workers,
                                               pin_memory=True, drop_last=True)

    valid_loader = torch.utils.data.DataLoader(valid_data,
                                               batch_size=config.batch_size,
                                               shuffle=False,
                                               num_workers=config.workers,
                                               pin_memory=True, drop_last=True)
    best_acc = 0.
    best_wa = 0.
    for epoch in range(config.epochs):
        train(model, optimizer, train_loader, epoch)
        cur_step = (epoch + 1) * len(train_loader)

        acc, wa = validate(valid_loader, model, epoch, cur_step, valid_data.get_class_weights())

        if best_acc < acc:
            best_acc = acc
            is_best = True
        else:
            is_best = False

        if best_wa < wa:
            best_wa = wa

        utils.save_checkpoint(model, config.path, is_best)

    logger.info("Final best Accuracy = {:.4%}".format(best_acc))


def train(model, optimizer, train_loader, epoch):
    losses = utils.AverageMeter()
    accuracy = utils.AverageMeter()

    cur_step = epoch * len(train_loader)

    for step, (trn_x, trn_y) in enumerate(train_loader):
        trn_x = trn_x.to(device)
        trn_y = torch.nn.functional.one_hot(trn_y, num_classes=4).float().to(device)
        N = trn_x.size(0)

        model.train()
        optimizer.zero_grad()

        output = model(trn_x)
        loss = torch.nn.functional.mse_loss(output, trn_y)
        loss.backward(retain_graph=True)

        optimizer.step()
        acc = utils.accuracy(output, trn_y)

        losses.update(loss.item(), N)
        accuracy.update(acc.item(), N)

        if step % config.print_freq == 0 or step == len(train_loader) - 1:
            logger.info("Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} Acc {accuracy.avg:.2%}".format(
                epoch + 1, config.epochs, step, len(train_loader) - 1, losses=losses, accuracy=accuracy
            ))

        writer.add_scalar('train/loss', loss.item(), cur_step)
        writer.add_scalar('train/acc', acc.item(), cur_step)
        cur_step += 1

    logger.info("Train: [{:2d}/{}] Final Acc {:.4%}".format(epoch + 1, config.epochs, accuracy.avg))


def validate(valid_loader, model, epoch, cur_step, class_weights):
    losses = utils.AverageMeter()
    accuracy = utils.AverageMeter()
    weighted_accuracy = utils.AverageMeter()
    f1s = utils.AverageMeter()
    recalls = utils.AverageMeter()
    precesions = utils.AverageMeter()

    model.eval()

    predictions = []
    true_labels = []

    with (torch.no_grad()):
        for step, (X, y) in enumerate(valid_loader):
            X = X.to(device, non_blocking=True)
            y = torch.nn.functional.one_hot(y, num_classes=4).float().to(device)
            N = X.size(0)

            logits = model(X)
            loss = torch.nn.functional.mse_loss(logits, y)
            losses.update(loss.item(), N)

            acc = utils.accuracy(logits, y)
            accuracy.update(acc.item(), N)

            wa, f1, r, p = utils.scores(logits, y, class_weights)

            weighted_accuracy.update(wa, N)
            f1s.update(f1, N)
            recalls.update(r, N)
            precesions.update(p, N)

            if step % config.print_freq == 0 or step == len(valid_loader) - 1:
                logger.info(
                    "Valid: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "Acc {accuracy.avg:.2%}".format(
                        epoch + 1, config.epochs, step, len(valid_loader) - 1, losses=losses,
                        accuracy=accuracy))

    writer.add_scalar('val/loss', losses.avg, cur_step)
    writer.add_scalar('val/acc', accuracy.avg, cur_step)
    writer.add_scalar('val/wa', weighted_accuracy.avg, cur_step)

    logger.info(
        "Valid: [{:2d}/{}] Final Acc {:.4%}".format(epoch + 1, config.epochs, accuracy.avg))

    return accuracy.avg, weighted_accuracy.avg


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
