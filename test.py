import numpy as np
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
from models.wrn import WideResNet
from utils.tools import create_logger
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# go through rigamaroo to do ...utils.display_results import show_performance
if __package__ is None:
    import sys
    from os import path

    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from utils.display_results import show_performance, get_measures, print_measures, print_measures_with_std
    import utils.svhn_loader as svhn
    import utils.lsun_loader as lsun_loader

parser = argparse.ArgumentParser(description='Evaluates a CIFAR OOD Detector',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Setup
parser.add_argument('--test_bs', type=int, default=200)
parser.add_argument('--num_to_avg', type=int, default=1, help='Average measures across num_to_avg runs.')
parser.add_argument('--validate', '-v', action='store_true', help='Evaluate performance on validation distributions.')
parser.add_argument('--method_name', '-m', type=str, default='cifar10_allconv_baseline', help='Method name.')
parser.add_argument('--queue-len', default=256, type=int, help='mmd-weight')
# Loading details
parser.add_argument('--layers', default=40, type=int, help='total number of layers')
parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')
parser.add_argument('--save', type=str, default=None, help='Checkpoint path to resume / test.')
parser.add_argument('--load', '-l', type=str, default=None, help='Checkpoint path to resume / test.')
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=4, help='Pre-fetching threads.')


def main():
    args = parser.parse_args()

    # create logger
    global logger
    logger = create_logger(
        os.path.join(args.save, 'logs', 'test_sfm.log'), 0)

    # mean and standard deviation of channels of CIFAR-10 images
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])

    if 'cifar10_' in args.method_name:
        test_data = dset.CIFAR10('data/cifarpy', train=False, transform=test_transform)
        num_classes = 10
    else:
        test_data = dset.CIFAR100('data/cifarpy', train=False, transform=test_transform)
        num_classes = 100


    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_bs, shuffle=False,
                                              num_workers=args.prefetch, pin_memory=True)

    # Create model
    logger.info("=> creating model '{}'".format(args.method_name))
    if 'wrn' in args.method_name:
        net = WideResNet(
            args.layers,
            num_classes,
            args.widen_factor, dropRate=args.droprate)

    # Restore model
    if args.load:
        checkpoint = torch.load(args.load)
        if 'state_dict' in checkpoint:
            net.load_state_dict(checkpoint['state_dict'])
        else:
            net.load_state_dict(checkpoint)
        logger.info("=> loaded checkpoint '{}'".format(args.load))

    # create csv
    csv_dir = os.path.join(args.save, 'test', 'sfm.csv')
    with open(csv_dir, 'w') as f:
        f.write('data,top1,fpr95,auroc,aupr\n')
        f.close()

    if args.ngpu > 1:
        net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

    if args.ngpu > 0:
        net.cuda()
        # torch.cuda.manual_seed(1)

    net.eval()

    cudnn.benchmark = True  # fire on all cylinders

    # /////////////// Detection Prelims ///////////////

    ood_num_examples = len(test_data) // 5
    expected_ap = ood_num_examples / (ood_num_examples + len(test_data))

    concat = lambda x: np.concatenate(x, axis=0)
    to_np = lambda x: x.data.cpu().numpy()

    def get_ood_scores(loader, in_dist=False):
        _score = []
        _right_score = []
        _wrong_score = []

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(loader):
                if batch_idx >= ood_num_examples // args.test_bs and in_dist is False:
                    break

                # forward
                data = data.cuda()
                output = net(data)
                _num_classes = output.size(-1)

                # get ood score
                smax = F.softmax(output, dim=1)
                _score.append(-np.max(to_np(smax), axis=1))

                if in_dist:
                    smax_in = to_np(F.softmax(output[:, :num_classes], dim=1))
                    preds = np.argmax(smax_in, axis=1)
                    targets = target.numpy().squeeze()
                    right_indices = preds == targets
                    wrong_indices = np.invert(right_indices)
                    _right_score.append(-np.max(smax_in[right_indices, :num_classes], axis=1))
                    _wrong_score.append(-np.max(smax_in[wrong_indices, :num_classes], axis=1))

        if in_dist:
            return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy()
        else:
            return concat(_score)[:ood_num_examples].copy()

    in_score, right_score, wrong_score = get_ood_scores(test_loader, in_dist=True)

    num_right = len(right_score)
    num_wrong = len(wrong_score)
    top1_err = num_wrong / (num_wrong + num_right)
    logger.info('=> * Error Rate {:.2f}'.format(100 * top1_err))

    # /////////////// End Detection Prelims ///////////////

    logger.info('=> Using CIFAR-10 as typical data') if num_classes == 10 else logger.info('=> Using CIFAR-100 as typical data')

    # /////////////// Error Detection ///////////////

    logger.info('=> Error Detection')
    show_performance(wrong_score, right_score, method_name=args.method_name, logger=logger)

    # /////////////// OOD Detection ///////////////
    auroc_list, aupr_list, fpr_list = [], [], []

    def get_and_print_results(ood_loader, num_to_avg=args.num_to_avg):

        aurocs, auprs, fprs = [], [], []
        for _ in range(num_to_avg):
            out_score = get_ood_scores(ood_loader)
            measures = get_measures(out_score, in_score)
            aurocs.append(measures[0]); auprs.append(measures[1]); fprs.append(measures[2])

        auroc = np.mean(aurocs); aupr = np.mean(auprs); fpr = np.mean(fprs)
        auroc_list.append(auroc); aupr_list.append(aupr); fpr_list.append(fpr)

        if num_to_avg >= 5:
            print_measures_with_std(aurocs, auprs, fprs, args.method_name, logger=logger)
        else:
            print_measures(auroc, aupr, fpr, args.method_name, logger=logger)
        return dict(fpr95=fpr, auroc=auroc, aupr=aupr)

    # /////////////// Textures ///////////////
    ood_data = dset.ImageFolder(root="data/dtd/images",
                                transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),
                                                       trn.ToTensor(), trn.Normalize(mean, std)]))
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                             num_workers=args.prefetch, pin_memory=True)

    logger.info('=> Texture Detection')
    results = get_and_print_results(ood_loader)
    with open(csv_dir, 'a') as f:
        f.write(f'Texture,{top1_err:02f},{results["fpr95"]:02f},{results["auroc"]:02f},{results["aupr"]:02f}\n')
        f.close()

    # /////////////// SVHN ///////////////
    ood_data = svhn.SVHN(root='data/svhn/', split="test",
                         transform=trn.Compose([trn.Resize(32), trn.ToTensor(), trn.Normalize(mean, std)]),
                         download=False)
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                             num_workers=args.prefetch, pin_memory=True)

    logger.info('=> SVHN Detection')
    results = get_and_print_results(ood_loader)
    with open(csv_dir, 'a') as f:
        f.write(f'SVHN,{top1_err:02f},{results["fpr95"]:02f},{results["auroc"]:02f},{results["aupr"]:02f}\n')
        f.close()

    # /////////////// Places365 ///////////////
    ood_data = dset.ImageFolder(root="data/place365/test",
                                transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),
                                                       trn.ToTensor(), trn.Normalize(mean, std)]))
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                             num_workers=args.prefetch, pin_memory=True)

    logger.info('=> Places365 Detection')
    results = get_and_print_results(ood_loader)
    with open(csv_dir, 'a') as f:
        f.write(f'Places365,{top1_err:02f},{results["fpr95"]:02f},{results["auroc"]:02f},{results["aupr"]:02f}\n')
        f.close()

    # /////////////// LSUN ///////////////
    ood_data = lsun_loader.LSUN("data/lsun", classes='test',
                                transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),
                                                       trn.ToTensor(), trn.Normalize(mean, std)]))
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                             num_workers=args.prefetch, pin_memory=True)

    logger.info('=> LSUN Detection')
    results = get_and_print_results(ood_loader)
    with open(csv_dir, 'a') as f:
        f.write(f'LSUN,{top1_err:02f},{results["fpr95"]:02f},{results["auroc"]:02f},{results["aupr"]:02f}\n')
        f.close()

    # /////////////// CIFAR Data ///////////////
    if 'cifar10_' in args.method_name:
        ood_data = dset.CIFAR100('data/cifarpy', train=False, transform=test_transform)
    else:
        ood_data = dset.CIFAR10('data/cifarpy', train=False, transform=test_transform)

    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                             num_workers=args.prefetch, pin_memory=True)

    cifar_name = 'CIFAR-10' if 'cifar100' in args.method_name else 'CIFAR-100'
    logger.info(f'=> {cifar_name} Detection')
    results = get_and_print_results(ood_loader)
    with open(csv_dir, 'a') as f:
        f.write(f'{cifar_name},{top1_err:02f},{results["fpr95"]:02f},{results["auroc"]:02f},{results["aupr"]:02f}\n')
        f.close()

    # /////////////// Mean Results ///////////////

    logger.info('=> Mean Test Results')
    print_measures(np.mean(auroc_list), np.mean(aupr_list), np.mean(fpr_list), method_name=args.method_name, logger=logger)
    with open(csv_dir, 'a') as f:
        f.write(f'Mean,{top1_err:02f},{np.mean(fpr_list):02f},{np.mean(auroc_list):02f},{np.mean(aupr_list):02f}\n')
        f.close()

if __name__ == '__main__':
    main()