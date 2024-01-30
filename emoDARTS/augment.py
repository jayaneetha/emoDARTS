""" Training augmented model """
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import utils
from config import AugmentConfig
from constant import DEVICE
from models.augment_cnn import AugmentCNN

config = AugmentConfig()

device = torch.device(DEVICE)

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

    # get data with meta info
    input_size, input_channels, n_classes, train_data, valid_data = utils.get_data(
        config.dataset, config.data_path, config.cutout_length, validation=True, features=config.features,
        fold=config.fold)

    dataset_label_weights = list(train_data.get_class_weights().values())

    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(dataset_label_weights).to(device)).to(device)
    use_aux = config.aux_weight > 0.
    model = AugmentCNN(input_size, input_channels, config.init_channels, n_classes, config.layers,
                       use_aux, config.genotype, n_layers_rnn=config.rnn_layers)

    if len(config.gpus) > 1:
        model = nn.DataParallel(model, device_ids=config.gpus).to(device)
    else:
        model = model.to(device)

    # model size
    mb_params = utils.param_size(model)
    logger.info("Model size = {:.3f} MB".format(mb_params))
    writer.add_text("Model Size:", "{:.3f} MB".format(mb_params))

    writer.add_text("Num Parameters:", "{}".format(utils.num_parameters(model)))

    # weights optimizer
    optimizer = torch.optim.SGD(model.parameters(), config.lr, momentum=config.momentum,
                                weight_decay=config.weight_decay)

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
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs)

    best_top1 = 0.
    best_acc = 0.
    best_wa = 0.
    # training loop
    for epoch in range(config.epochs):
        lr_scheduler.step()
        drop_prob = config.drop_path_prob * epoch / config.epochs
        # model.module.drop_path_prob(drop_prob)
        model.drop_path_prob(drop_prob)

        # training
        train(train_loader, model, optimizer, criterion, epoch)

        # validation
        cur_step = (epoch + 1) * len(train_loader)
        top1, acc, wa = validate(valid_loader, model, criterion, epoch, cur_step, valid_data.get_class_weights())

        # save
        if best_top1 < top1:
            best_top1 = top1
            is_best = True
        else:
            is_best = False

        if best_acc < acc:
            best_acc = acc

        if best_wa < wa:
            best_wa = wa

        utils.save_checkpoint(model, config.path, is_best)

        print("")

    logger.info("Final best Prec@1 = {:.4%}".format(best_top1))
    writer.add_text("Final Best Prec@1", "{:.4%}".format(best_top1))

    logger.info("Final best Acc = {:.4%}".format(best_acc))
    writer.add_text("Final Best Acc", "{:.4%}".format(best_acc))
    writer.add_text("Final Best Weighted Acc", "{:.4%}".format(best_wa))


def train(train_loader, model, optimizer, criterion, epoch):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()
    accuracy = utils.AverageMeter()

    cur_step = epoch * len(train_loader)
    cur_lr = optimizer.param_groups[0]['lr']
    logger.info("Epoch {} LR {}".format(epoch, cur_lr))
    writer.add_scalar('train/lr', cur_lr, cur_step)

    with torch.autograd.set_detect_anomaly(True):

        model.train()

        for step, (X, y) in enumerate(train_loader):
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            N = X.size(0)

            optimizer.zero_grad()
            logits, aux_logits = model(X)
            loss = criterion(logits, y)
            if config.aux_weight > 0.:
                loss += config.aux_weight * criterion(aux_logits, y)
            loss.backward()
            # gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            prec1, prec5 = utils.precision(logits, y, topk=(1, 4))
            losses.update(loss.item(), N)
            top1.update(prec1.item(), N)
            top5.update(prec5.item(), N)

            acc = utils.accuracy(logits, y)
            accuracy.update(acc.item(), N)

            if step % config.print_freq == 0 or step == len(train_loader) - 1:
                logger.info(
                    "Train: [{:3d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%}) "
                    "Acc {accuracy.avg:.2%}".format(
                        epoch + 1, config.epochs, step, len(train_loader) - 1, losses=losses,
                        top1=top1, top5=top5, accuracy=accuracy))

            writer.add_scalar('train/loss', loss.item(), cur_step)
            writer.add_scalar('train/top1', prec1.item(), cur_step)
            writer.add_scalar('train/acc', acc.item(), cur_step)
            cur_step += 1

        logger.info("Train: [{:3d}/{}] Final Prec@1 {:.4%} Final Acc {:.4%}".format(epoch + 1, config.epochs, top1.avg,
                                                                                    accuracy.avg))


def validate(valid_loader, model, criterion, epoch, cur_step, class_weights):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()
    accuracy = utils.AverageMeter()
    weighted_accuracy = utils.AverageMeter()
    f1s = utils.AverageMeter()
    recalls = utils.AverageMeter()
    precesions = utils.AverageMeter()

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

            wa, f1, r, p = utils.scores(logits, y, class_weights)

            weighted_accuracy.update(wa, N)
            f1s.update(f1, N)
            recalls.update(r, N)
            precesions.update(p, N)

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
    writer.add_scalar('val/wa', weighted_accuracy.avg, cur_step)

    logger.info("Valid: [{:3d}/{}] Final Prec@1 {:.4%} Final Acc {:.4%} Final Weighted Acc {:.4%}".format(epoch + 1,
                                                                                                          config.epochs,
                                                                                                          top1.avg,
                                                                                                          accuracy.avg,
                                                                                                          weighted_accuracy.avg))
    return top1.avg, accuracy.avg, weighted_accuracy.avg


if __name__ == "__main__":
    main()
