import os
import argparse

import numpy as np

parser = argparse.ArgumentParser(description='Evaluates a CIFAR OOD Detector',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--save', '-s', type=str, default=None, help='path to save results.')
parser.add_argument('--test_dir', type=str, default='test', help='path to save results.')
parser.add_argument('--csv_file', type=str, default='results', help='path to save results.')
parser.add_argument('--seeds', type=int, nargs="+", default=None, help='seeds')
args = parser.parse_args()

results = dict()
for seed in args.seeds:
    csv_file = f'{args.save}/seed_{seed}/test/{args.csv_file}.csv'
    with open(csv_file, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if i == 0:
                continue
            data, top1, fpr, auroc, aupr = line.strip('\n').split(',')
            if data not in results:
                results[data] = dict(top1=[], fpr=[], auroc=[], aupr=[])
            results[data]['top1'].append(float(top1))
            results[data]['fpr'].append(float(fpr))
            results[data]['auroc'].append(float(auroc))
            results[data]['aupr'].append(float(aupr))
        f.close()

str_bak = ''
for i, data in enumerate(results.keys()):
    top1s = np.array(results[data]['top1'])
    fprs = np.array(results[data]['fpr'])
    aurocs = np.array(results[data]['auroc'])
    auprs = np.array(results[data]['aupr'])
    print(f'{data} Detection')
    print('\t\t\t\t' + os.path.basename(args.save))
    print(f'TOP1: \t\t\t{100 * np.mean(top1s):.2f}\t+/- {100 * np.std(top1s):.2f}')
    print(f'FPR95:\t\t\t{100 * np.mean(fprs):.2f}\t+/- {100 * np.std(fprs):.2f}')
    print(f'AUROC:\t\t\t{100 * np.mean(aurocs):.2f}\t+/- {100 * np.std(aurocs):.2f}')
    print(f'AUPR: \t\t\t{100 * np.mean(auprs):.2f}\t+/- {100 * np.std(auprs):.2f}\n')
    top1_str = f'{100 * np.mean(top1s):.2f} +/- {100 * np.std(top1s):.2f}'
    fpr_str = f'{100 * np.mean(fprs):.2f} +/- {100 * np.std(fprs):.2f}'
    auroc_str = f'{100 * np.mean(aurocs):.2f} +/- {100 * np.std(aurocs):.2f}'
    aupr_str = f'{100 * np.mean(auprs):.2f} +/- {100 * np.std(auprs):.2f}'
    if data == 'Mean':
        str_bak = f'{top1_str},{fpr_str},{auroc_str},{aupr_str},' + str_bak
    else:
        str_bak += f'{top1_str},{fpr_str},{auroc_str},{aupr_str},'

csv_file = f'{args.save}/{args.test_dir}/{os.path.basename(args.save)}_{args.csv_file}.csv'
with open(csv_file, 'w') as f:
    f.write(str_bak)
    f.close()
