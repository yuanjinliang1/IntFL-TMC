import argparse

def parse_args():
    model_names = ['lenet','vgg']
    parser = argparse.ArgumentParser(description='PyTorch integer MNIST Example')
    parser.add_argument('--model-type', default='float',choices=['int','float','hybrid'],
                        help='choose to train int model or float model')
    parser.add_argument('--model', '-a', metavar='MODEL', default='vgg',choices=model_names,
                        help='model architecture: ' +' | '.join(model_names))
    parser.add_argument('--dataset', metavar='DATASET', default='cifar10',
                        help='dataset name or folder')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    # parser.add_argument('--log-interval', type=int, default=10, metavar='N',
    #                     help='how many batches to wait before logging training status')
    # parser.add_argument('--num-round', default=2000, type=int, metavar='N',
    #                     help='number of total round to run')
    # parser.add_argument('--num-epochs', default=5, type=int, metavar='N',
    #                     help='number of epoch to run in each client')
    # parser.add_argument('--num-clients', default=20, type=int, metavar='N',
    #                     help='number of selected clients to run in each round')
    parser.add_argument('--log-interval', type=int, default=30, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--num-round', default=4000, type=int, metavar='N',
                        help='number of total round to run')
    parser.add_argument('--num-epochs', default=20, type=int, metavar='N',
                        help='number of epoch to run in each client')
    parser.add_argument('--num-clients', default=50, type=int, metavar='N',
                        help='number of selected clients to run in each round')
    parser.add_argument('--total-clients', default=500, type=int, metavar='N',
                        help='number of selected clients to run in each round')
    parser.add_argument('--save-all', action='store_true', default=False,
                        help='Save all checkpoints along training')
    parser.add_argument('--weight-frac', action='store_true', default=False,
                        help='add another 8 bits fraction for gradient accumulation')
    parser.add_argument('--weight-decay', action='store_true', default=False,
                        help='integer training weight decay')
    parser.add_argument('--weight-hist', action='store_true', default=True,
                        help='record weight histogram after each epoch finishes')
    parser.add_argument('--grad-hist', action='store_true', default=True,
                        help='record gradient histogram during training')
    parser.add_argument('--download', action='store_true', default=False,
                        help='Download dataset')
    parser.add_argument('--results-dir', default='./results --save test', metavar='RESULTS_DIR',
                        help='results dir')
    parser.add_argument('--data-dir', default='/niti', 
                        help='dataset dir')
    parser.add_argument('--save', metavar='SAVE', default='',
                        help='saved folder')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
    parser.add_argument('--depth',action="store",dest='depth',default=16,type=int,
                        help='resnet depth')
    parser.add_argument('--seed', type=int, default=-1, metavar='S',
                        help='random seed (default: None)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-e', '--evaluate', type=str, metavar='FILE',
                        help='evaluate model FILE on validation set')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--init', default='', type=str, metavar='PATH',
                        help='path to weight init checkpoint (default: none)')
    
    return parser.parse_args()
