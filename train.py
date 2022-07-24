# -*- coding: utf-8 -*-
import os
import shutil
import argparse
import time
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F

from models.wrn import WideResNet
from utils.tools import (create_logger, SummaryWriter,
                         restart_from_checkpoint,
                         AverageMeter, PaceAverageMeter)
from utils.hsic import RbfHSIC
from utils.mmd import MMD
from utils.rand_augment import RandAugment

if __package__ is None:
    import sys
    from os import path

    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from utils.tinyimages_80mn_loader import TinyImages

parser = argparse.ArgumentParser(description='Trains a CIFAR Classifier with OE',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('dataset', type=str, choices=['cifar100'],
                    help='Choose between CIFAR-100.')
parser.add_argument('--model', '-m', type=str, default='allconv',
                    choices=['allconv', 'wrn'], help='Choose architecture.')
# Optimization options
parser.add_argument('--epochs', '-e', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, help='The initial learning rate.')
parser.add_argument('--batch_size', '-b', type=int, default=128, help='Batch size.')
parser.add_argument('--oe_batch_size', type=int, default=256, help='Batch size.')
parser.add_argument('--test_bs', type=int, default=200)
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
# WRN Architecture
parser.add_argument('--layers', default=40, type=int, help='total number of layers')
parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')
# Loss weight
parser.add_argument('--oe-weight', default=0., type=float, help='ood loss weight')
parser.add_argument('--mmd-weight', default=0., type=float, help='mmd-weight')
parser.add_argument('--hsic-scale', default=16384, type=float, help='mmd-weight')
parser.add_argument('--hsic-weight', default=0., type=float, help='mmd-weight')
parser.add_argument('--hsic-tau', default=3., type=float, help='hsic tau')
# Checkpoints
parser.add_argument('--save', '-s', type=str, default='./snapshots/oe_scratch', help='Folder to save checkpoints.')
parser.add_argument('--resume', '-l', type=str, default=None, help='Checkpoint path to resume / test.')
parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')
parser.add_argument('--ckp-freq', type=int, default=10, help='Save the model periodically')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=4, help='Pre-fetching threads.')
# Random seed
parser.add_argument("--seed", type=int, default=0, help="seed")
parser.add_argument("--disable_random", type=int, default=0, help="disable_random")
# RandAugment
parser.add_argument("--aug", type=int, default=0, help="use RandAugment")
parser.add_argument("--aug-n", type=int, default=2, help="RandAugment: N")
parser.add_argument("--aug-m", type=int, default=10, help="RandAugment: M")


def main():
    args = parser.parse_args()

    # Make save directory
    os.makedirs(os.path.join(args.save, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(args.save, 'checkpoints'), exist_ok=True)

    state = {k: v for k, v in args._get_kwargs()}

    # create logger and tensorboard writer
    global logger
    logger = create_logger(
        os.path.join(args.save, 'logs', 'train.log'), 0)
    global writer
    writer = SummaryWriter(args.save)
    logger.info(state)

    # set seed
    if args.seed:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)

    # mean and standard deviation of channels of CIFAR-10 images
    mean = [x / 255. for x in [125.3, 123.0, 113.9]]
    std = [x / 255. for x in [63.0, 62.1, 66.7]]

    # Create training and val dataset
    train_transform = trn.Compose([
        trn.RandomHorizontalFlip(),
        trn.RandomCrop(32, padding=4),
        trn.ToTensor(),
        trn.Normalize(mean, std)])

    test_transform = trn.Compose([
        trn.ToTensor(),
        trn.Normalize(mean, std)])

    train_data_in = dset.CIFAR100('data/cifarpy', train=True, transform=train_transform)
    test_data = dset.CIFAR100('data/cifarpy', train=False, transform=test_transform)
    num_classes = 100
    args.num_classes = num_classes
    logger.info(f"=> Load inlier data ({args.dataset}): {len(train_data_in)} images")

    # Create ood dataset
    if args.aug:
        ood_data = TinyImages(
            transform=trn.Compose([
                trn.ToTensor(),
                trn.ToPILImage(),
                trn.RandomCrop(32, padding=4),
                RandAugment(n=args.aug_n, m=args.aug_m),
                trn.RandomHorizontalFlip(),
                trn.ToTensor(),
                trn.Normalize(mean, std)]))
    else:
        ood_data = TinyImages(
            transform=trn.Compose([
                trn.ToTensor(),
                trn.ToPILImage(),
                trn.RandomCrop(32, padding=4),
                trn.RandomHorizontalFlip(),
                trn.ToTensor(),
                trn.Normalize(mean, std)]))
    logger.info(f"=> Load outlier data (TinyImages): {len(ood_data)} images")

    # Create dataloader
    train_loader_in = torch.utils.data.DataLoader(
        train_data_in,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.prefetch, pin_memory=True, drop_last=True)

    train_loader_ood = torch.utils.data.DataLoader(
        ood_data,
        batch_size=args.oe_batch_size, shuffle=False,
        num_workers=args.prefetch, pin_memory=True, sampler=None)

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.prefetch, pin_memory=True, drop_last=True)

    # Create model
    logger.info("=> creating model '{}'".format(args.model))
    if args.model == 'wrn':
        net = WideResNet(
            args.layers,
            num_classes,
            args.widen_factor,
            dropRate=args.droprate)

    if args.ngpu > 1:
        net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

    if args.ngpu > 0:
        net.cuda()
        torch.cuda.manual_seed(args.seed)

    cudnn.benchmark = True  # fire on all cylinders

    optimizer = torch.optim.SGD(
        net.parameters(), state['learning_rate'], momentum=state['momentum'],
        weight_decay=state['decay'], nesterov=True)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_annealing(
            step,
            args.epochs * len(train_loader_in),
            1,  # since lr_lambda computes multiplicative factor
            1e-6 / args.learning_rate))

    # Restore model if desired
    to_restore = {"epoch": 0}
    resume_path = os.path.join(args.save, "checkpoint.pth.tar")
    if args.resume is not None:
        resume_path = args.resume
    restart_from_checkpoint(
        resume_path,
        run_variables=to_restore,
        state_dict=net,
        optimizer=optimizer,
        scheduler=scheduler)
    start_epoch = to_restore["epoch"]

    if args.test:
        test()
        print(state)
        exit()

    logger.info('=> Beginning training')

    # Main loop
    for epoch in range(start_epoch, args.epochs):
        state['epoch'] = epoch

        train(train_loader_in, train_loader_ood, net, optimizer, scheduler, state, args)
        test(test_loader, net, state)

        # Save model
        save_dict = {
            'epoch': epoch + 1,
            'model': args.model,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }
        torch.save(save_dict, os.path.join(args.save, "checkpoint.pth.tar"))
        if (epoch + 1) % args.ckp_freq == 0 or (epoch + 1) == args.epochs:
            shutil.copyfile(
                os.path.join(args.save, "checkpoint.pth.tar"),
                os.path.join(args.save, 'checkpoints', "ckp-" + str(epoch) + ".pth"),
            )


# train function
def train(train_loader_in, train_loader_ood, net, optimizer, scheduler, state, args):
    net.train()  # enter train mode
    epoch = state['epoch']
    sample_number = len(train_loader_in)

    left_time = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    ce_losses = PaceAverageMeter(pace=200)
    if args.oe_weight > 0.:
        oe_losses = PaceAverageMeter(pace=200)
    mmd = MMD()
    if args.mmd_weight > 0.:
        mmd_losses = PaceAverageMeter(pace=200)
    hsic_rbf = RbfHSIC(sigma_x=args.hsic_tau)
    if args.hsic_weight > 0.:
        hsic_losses = PaceAverageMeter(pace=200)
    losses = PaceAverageMeter(pace=200)
    top1 = PaceAverageMeter(pace=200)

    if args.disable_random:
        # fixed version for removing randomness of training ood data
        train_loader_ood.dataset.offset = int(len(train_loader_ood.dataset) * 1. / args.epochs * epoch)
    else:
        # OE random version
        train_loader_ood.dataset.offset = np.random.randint(len(train_loader_ood.dataset))

    end = time.time()
    for it, (in_set, out_set) in enumerate(zip(train_loader_in, train_loader_ood)):
        data_time.update(time.time() - end)
        data = torch.cat((in_set[0], out_set[0]), 0)
        target = in_set[1]
        in_bs = target.size(0)

        data, target = data.cuda(), target.cuda()

        # forward
        x, feat = net(data, return_feat=True)
        in_feat = feat[:in_bs]
        ood_feat = feat[in_bs:]

        # cross entropy loss
        ce_loss = F.cross_entropy(x[:in_bs], target)
        loss = ce_loss
        ce_losses.update(ce_loss.item(), data.size(0))

        # oe loss
        if args.oe_weight > 0.:
            oe_loss = args.oe_weight * -(x[in_bs:].mean(1) - torch.logsumexp(x[in_bs:], dim=1)).mean()
            oe_losses.update(oe_loss.item(), data.size(0))
            loss += oe_loss

        # mmd loss
        if args.mmd_weight > 0.:
            mmd_loss = args.mmd_weight * -mmd([ood_feat], [in_feat])
            mmd_losses.update(mmd_loss.item(), data.size(0))
            loss += mmd_loss

        # hsic loss
        if args.hsic_weight > 0.:
            hsic_loss = hsic_rbf(ood_feat[:in_bs], in_feat) + hsic_rbf(ood_feat[in_bs:], in_feat)
            hsic_loss *= args.hsic_scale * args.hsic_weight / 2.
            hsic_losses.update(hsic_loss.item(), data.size(0))
            loss += hsic_loss

        acc1, acc5 = accuracy(x[:len(in_set[0]), :args.num_classes], target, topk=(1, 5))
        losses.update(loss.item(), data.size(0))
        top1.update(acc1[0], data.size(0))

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        left_time.update(batch_time.val * ((args.epochs - epoch) * len(train_loader_in) - it) / 3600.0)
        end = time.time()

        if (it + 1) % 50 == 0:
            log_str = ''
            log_str += f"Epoch: [{epoch}][{it + 1}/{sample_number}]\t"
            log_str += f"Left Time {left_time.val:.3f} ({left_time.avg:.3f})\t"
            log_str += f"Loss {losses.val:.4f} ({losses.avg:.4f})\t"
            log_str += f"CE Loss {ce_losses.val:.4f} ({ce_losses.avg:.4f})\t"
            if args.oe_weight > 0.:
                log_str += f"OE Loss {oe_losses.val:.4f} ({oe_losses.avg:.4f})\t"
            if args.mmd_weight > 0.:
                log_str += f"MMD Loss {mmd_losses.val:.4f} ({mmd_losses.avg:.4f})\t"
            if args.hsic_weight > 0.:
                log_str += f"HSIC Loss {hsic_losses.val:.4f} ({hsic_losses.avg:.4f})\t"
            log_str += f"Error@1 {(1. - top1.val) * 100:.4f} ({(1. - top1.avg) * 100:.4f})\t"
            log_str += f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
            log_str += f"Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
            log_str += f"Lr: {optimizer.param_groups[0]['lr']:.4f}"
            logger.info(log_str)
            total_it = epoch * sample_number + it
            writer.log(iter=total_it, tag='train/loss', val=losses.avg)
            writer.log(iter=total_it, tag='train/ce_loss', val=ce_losses.avg)
            if args.oe_weight > 0.:
                writer.log(iter=total_it, tag='train/oe_loss', val=oe_losses.avg)
            if args.mmd_weight > 0.:
                writer.log(iter=total_it, tag='train/mmd_loss', val=mmd_losses.avg)
            if args.hsic_weight > 0.:
                writer.log(iter=total_it, tag='train/hsic_loss', val=hsic_losses.avg)
            writer.log(iter=total_it, tag='train/err@1', val=1. - top1.avg)
            writer.log(iter=total_it, tag='train/lr', val=optimizer.param_groups[0]["lr"])


# test function
def test(test_loader, net, state):
    net.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()

            # forward
            output = net(data)

            # accuracy
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).sum().item()

    test_accuracy = correct / len(test_loader.dataset)
    logger.info(f'Eval: * Error@1 {(1. - test_accuracy) * 100.:.3f}')
    writer.log(iter=state['epoch'], tag='eval/err@1', val=(1. - test_accuracy) * 100.)


def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
            1 + np.cos(step / total_steps * np.pi))


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
