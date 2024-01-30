""" Training augmented model """
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter

import utils
from config import ValidateConfig
from constant import DEVICE
from models.augment_cnn import AugmentCNN

config = ValidateConfig()

device = torch.device(DEVICE)

# tensorboard
writer = SummaryWriter(log_dir=os.path.join(config.path, "tb"))
writer.add_text('config', config.as_markdown(), 0)

logger = utils.get_logger(os.path.join(config.path, "{}.log".format(config.name)))
config.print_params(logger.info)


def main():
    logger.info("Logger is set - validation start")

    # set default gpu device id
    torch.cuda.set_device(config.gpus[0])

    # set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    torch.backends.cudnn.benchmark = True

    # get data with meta info
    input_size, input_channels, n_classes, train_data, valid_data = utils.get_data(
        config.dataset, config.data_path, config.cutout_length, validation=True, features=config.features,
        fold=config.fold)

    criterion = nn.CrossEntropyLoss().to(device)
    use_aux = config.aux_weight > 0.
    model = AugmentCNN(input_size, input_channels, config.init_channels, n_classes, config.layers,
                       use_aux, config.genotype, n_layers_rnn=config.rnn_layers)

    model = torch.load(config.model_file)

    if len(config.gpus) > 1:
        model = nn.DataParallel(model, device_ids=config.gpus).to(device)
    else:
        model = model.to(device)

    # model.load_state_dict(load['state_dict'])

    # model size
    mb_params = utils.param_size(model)
    logger.info("Model size = {:.3f} MB".format(mb_params))
    writer.add_text("Model Size:", "{:.3f} MB".format(mb_params))

    writer.add_text("Num Parameters:", "{}".format(utils.num_parameters(model)))

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=config.batch_size,
                                               shuffle=True,
                                               num_workers=config.workers,
                                               pin_memory=True, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(valid_data,
                                               batch_size=config.batch_size,
                                               shuffle=False,
                                               num_workers=config.workers,
                                               pin_memory=True, drop_last=True)

    best_top1 = 0.
    best_acc = 0.
    best_conf_matrix = None
    # training loop
    for epoch in range(config.epochs):
        drop_prob = config.drop_path_prob * epoch / config.epochs
        # model.module.drop_path_prob(drop_prob)
        model.drop_path_prob(drop_prob)

        # validation
        cur_step = (epoch + 1) * len(train_loader)
        top1, acc, predictions, true_labels, conf_matrix = validate(valid_loader, model, criterion, epoch, cur_step)

        # save
        if best_top1 < top1:
            best_top1 = top1
            is_best = True
        else:
            is_best = False

        if best_acc < acc:
            best_acc = acc

        if is_best:
            best_conf_matrix = conf_matrix

        print("")

    logger.info("Final best Prec@1 = {:.4%}".format(best_top1))
    writer.add_text("Final Best Prec@1", "{:.4%}".format(best_top1))

    logger.info("Final best Acc = {:.4%}".format(best_acc))
    writer.add_text("Final Best Acc", "{:.4%}".format(best_acc))


def validate(valid_loader, model, criterion, epoch, cur_step):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()
    accuracy = utils.AverageMeter()

    model.eval()

    predictions = []
    true_labels = []

    with torch.no_grad():
        for step, (X, y) in enumerate(valid_loader):
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            N = X.size(0)

            logits, _ = model(X)
            loss = criterion(logits, y)

            predictions.extend(logits.cpu().numpy())
            true_labels.extend(y.cpu().numpy())

            prec1, prec5 = utils.precision(logits, y, topk=(1, 4))
            losses.update(loss.item(), N)
            top1.update(prec1.item(), N)
            top5.update(prec5.item(), N)

            acc = utils.accuracy(logits, y)
            accuracy.update(acc.item(), N)

            if step % config.print_freq == 0 or step == len(valid_loader) - 1:
                logger.info(
                    "Valid: [{:3d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%}) "
                    "Acc {accuracy.avg:.2%}".format(
                        epoch + 1, config.epochs, step, len(valid_loader) - 1, losses=losses,
                        top1=top1, top5=top5, accuracy=accuracy))

    writer.add_scalar('val/loss', losses.avg, cur_step)
    writer.add_scalar('val/top1', top1.avg, cur_step)
    writer.add_scalar('val/acc', accuracy.avg, cur_step)

    logger.info("Valid: [{:3d}/{}] Final Prec@1 {:.4%} Final Acc {:.4%}".format(epoch + 1, config.epochs, top1.avg,
                                                                                accuracy.avg))
    conf_matrix = confusion_matrix(true_labels, np.argmax(predictions, axis=1))

    return top1.avg, accuracy.avg, predictions, true_labels, conf_matrix


if __name__ == "__main__":
    main()
