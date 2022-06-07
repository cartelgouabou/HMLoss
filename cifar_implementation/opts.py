import argparse
import models



parser = argparse.ArgumentParser(description='PyTorch Cifar')
parser.add_argument('--dataset', default='cifar10', help='dataset setting')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet32')
parser.add_argument('--loss_type', default="CE", type=str, help='loss type')
parser.add_argument('--delta', default=10000, type=int, help='delta parameter for HMLoss')
parser.add_argument('--imb_type', default="exp", type=str, help='imbalance type')
parser.add_argument('--imb_factor', default=0.02, type=float, help='imbalance factor')
parser.add_argument('--imb_ratio_list','--ir_list', nargs='+',metavar='N', type=float, default=0.01, help='imbalance ratio, 1 for balanced, 0.1 or 0.01; use case: --ir 1 0.1 0.01 0.02',dest='ir_list')
parser.add_argument('--weighting_type', default='None', type=str, help='data sampling strategy for train loader')
parser.add_argument('--start_hm_epoch', default=100, type=int, help='start epoch for HM Loss')
parser.add_argument('--rand_number', default=0, type=int, help='fix random number for data sampling')
parser.add_argument('--num_runs', default=10, type=int, help='number of runs to launch ') #5
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run') 
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=2e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print_freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 100)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--path_history',type=str, default='history')
parser.add_argument('--path_model', type=str, default='checkpoint')




