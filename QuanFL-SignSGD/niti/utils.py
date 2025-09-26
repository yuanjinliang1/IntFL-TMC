import torch
import torchvision.transforms as transforms
import torch.utils.data as Data
# import torch.utils.data as data
import torch.nn.functional as F
from torch.autograd import Variable

from collections import defaultdict
from PIL import Image
import os
import json
import numpy as np

import re

from args import parse_args
from preprocess import get_transform
from datasets import CIFAR10_truncated

global args
args = parse_args()

__imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                   'std': [0.229, 0.224, 0.225]}


'''
preprocess reference : https://github.com/google-research/federated/blob/master/utils/datasets/cifar100_dataset.py
'''

def cifar100_transform(img_mean, img_std, train = True, crop_size = (24,24)):
    """cropping, flipping, and normalizing."""
    if train:
        return transforms.Compose([
            transforms.ToPILImage(),
            # transforms.RandomCrop(crop_size),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize(mean=img_mean, std=img_std),
            transforms.Normalize(mean=[n/255.
                        for n in [129.3, 124.1, 112.4]], std=[n/255. for n in [68.2,  65.4,  70.4]]),
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            # transforms.CenterCrop(crop_size),
            # transforms.CenterCrop(32, padding=4),
            transforms.ToTensor(),
            # transforms.Normalize(mean=img_mean, std=img_std),
            transforms.Normalize(mean=[n/255.
                        for n in [129.3, 124.1, 112.4]], std=[n/255. for n in [68.2,  65.4,  70.4]]),
        ])


def preprocess_cifar_img(img, train):
    # scale img to range [0,1] to fit ToTensor api
    img = torch.div(img, 255.0)
    # for i in img:
    #     i = i.permute(2,0,1)
    #     mean = torch.mean(i)
    #     std = i.std()
    #     print()
    transoformed_img = torch.stack([cifar100_transform
        (i.type(torch.DoubleTensor).mean(),
            i.type(torch.DoubleTensor).std(),
            train)
        (i.permute(2,0,1)) 
        for i in img])
    return transoformed_img

def read_data(train_data_dir, test_data_dir):
    train_clients, train_data = read_dir(train_data_dir)
    test_clients, test_data = read_dir(test_data_dir)
    return train_clients, train_data, test_data

def read_dir(data_dir):
    clients = []
    data = defaultdict(lambda : None)
    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        if args.dataset == 'cifar10':
            user = re.findall(r"\d+", f)[0]
            clients.append(user)
            cdata_dict = {}
            cdata_dict[user] = cdata
            data.update(cdata_dict)
        else:
            clients.extend(cdata['users'])
            data.update(cdata['user_data'])
        
    clients = list(sorted(data.keys()))
    return clients, data

## leaf data to dataloader
def utils_loader(data, batch_size, train_or_test):
    if args.dataset == 'mnist':
        x = torch.reshape(torch.tensor(data['x']), (-1, 1, 28, 28))
        y = torch.tensor(data['y'], dtype=torch.int64)
    elif args.dataset == 'cifar10':
        # x = torch.reshape(torch.tensor(data['x'], dtype=torch.float), (-1, 1, 32, 32))
        x = torch.tensor(data['x'], dtype=torch.float)
        # transform_train = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(**__imagenet_stats),
        # ])
        # x = transform_train(np.array(data['x']))
        y = torch.tensor(data['y'], dtype=torch.int64)
    elif args.dataset == 'celeba':
        x = [load_image(i) for i in data['x']]
        x = torch.reshape(torch.tensor(np.array(x), dtype=torch.float), (-1, 3, 84, 84))
        x = F.normalize(x, p=2, dim=1)
        # y = torch.tensor(np.array(data['y']), dtype=torch.int64)
        y = torch.tensor(np.array(data['y']), dtype=torch.int64)
    if train_or_test == 'train':
        batch_size = batch_size
    elif train_or_test == 'test':
        batch_size = len(y)
    torch_dataset = Data.TensorDataset(x, y)
    data_leaf_loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    return data_leaf_loader

def load_image(img_name):
    IMAGES_DIR = '../fed_data/leaf/celeba/data/raw/img_align_celeba'
    img = Image.open(os.path.join(IMAGES_DIR, img_name))
    img = img.resize((84, 84)).convert('RGB')
    return np.array(img)


def partition_data(dataset, datadir, n_nets, alpha):
    '''
    partition cifar10 data to FedCifar10 dataset

    Args:
         dataset: origin cifar10 dataset
         n_nets: total number of client after federated partition
         alpha: repetitive rate of data samples in each partitioned clients  
    '''
    if dataset == 'cifar10':
        X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
        n_train = X_train.shape[0]

        min_size = 0
        K = 10
        N = y_train.shape[0]
        net_dataidx_map = {}

        while min_size < 10:
            idx_batch = [[] for _ in range(n_nets)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
                ## Balance
                proportions = np.array([p*(len(idx_j)<N/n_nets) for p,idx_j in zip(proportions,idx_batch)])
                proportions = proportions/proportions.sum()
                proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

        # N_test = y_test.shape[0]
        # net_dataidx_map_test = {}

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)

    #return (X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts)
    return y_train, net_dataidx_map, traindata_cls_counts


def load_cifar10_data(datadir):

    transform = transforms.Compose([transforms.ToTensor()])

    cifar10_train_ds = CIFAR10_truncated(datadir, train=True, download=True, transform=transform)
    cifar10_test_ds = CIFAR10_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    return (X_train, y_train, X_test, y_test)


def record_net_data_stats(y_train, net_dataidx_map):

    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    # logging.debug('Data statistics: %s' % str(net_cls_counts))
    return net_cls_counts


def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None):
    '''
    constract dataloader

    Args:
        train_bs: train batch size
    '''
    if dataset in ('mnist', 'cifar10'):
        if dataset == 'mnist':
            # dl_obj = MNIST_truncated

            transform_train = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))])

            transform_test = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))])

        elif dataset == 'cifar10':
            dl_obj = CIFAR10_truncated

            normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                std=[x/255.0 for x in [63.0, 62.1, 66.7]])
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: F.pad(
                                    Variable(x.unsqueeze(0), requires_grad=False),
                                    (4,4,4,4),mode='reflect').data.squeeze()),
                transforms.ToPILImage(),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
                ])
            # data prep for test set
            transform_test = transforms.Compose([transforms.ToTensor(),normalize])


        train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True)
        test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True)

        train_dl = Data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True)
        test_dl = Data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False)

    return train_dl, test_dl


# import torch
# import torchvision.transforms as transforms
# import torch.utils.data as Data
# # import torch.utils.data as data
# import torch.nn.functional as F
# from torch.autograd import Variable

# from collections import defaultdict
# from PIL import Image
# import os
# import json
# import numpy as np

# import re

# from args import parse_args
# from preprocess import get_transform
# from datasets import CIFAR10_truncated

# global args
# args = parse_args()

# __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
#                    'std': [0.229, 0.224, 0.225]}


# '''
# preprocess reference : https://github.com/google-research/federated/blob/master/utils/datasets/cifar100_dataset.py
# '''

# def cifar100_transform(img_mean, img_std, train = True, crop_size = (24,24)):
#     """cropping, flipping, and normalizing."""
#     if train:
#         return transforms.Compose([
#             transforms.ToPILImage(),
#             transforms.RandomCrop(crop_size),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=img_mean, std=img_std),
#         ])
#     else:
#         return transforms.Compose([
#             transforms.ToPILImage(),
#             transforms.CenterCrop(crop_size),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=img_mean, std=img_std),
#         ])


# def preprocess_cifar_img(img, train):
#     # scale img to range [0,1] to fit ToTensor api
#     img = torch.div(img, 255.0)
#     transoformed_img = torch.stack([cifar100_transform
#         (i.type(torch.DoubleTensor).mean(),
#             i.type(torch.DoubleTensor).std(),
#             train)
#         (i.permute(2,0,1)) 
#         for i in img])
#     return transoformed_img

# def read_data(train_data_dir, test_data_dir):
#     train_clients, train_data = read_dir(train_data_dir)
#     test_clients, test_data = read_dir(test_data_dir)
#     return train_clients, train_data, test_data

# def read_dir(data_dir):
#     clients = []
#     data = defaultdict(lambda : None)
#     files = os.listdir(data_dir)
#     files = [f for f in files if f.endswith('.json')]
#     for f in files:
#         file_path = os.path.join(data_dir,f)
#         with open(file_path, 'r') as inf:
#             cdata = json.load(inf)
#         if args.dataset == 'cifar10':
#             user = re.findall(r"\d+", f)[0]
#             clients.append(user)
#             cdata_dict = {}
#             cdata_dict[user] = cdata
#             data.update(cdata_dict)
#         else:
#             clients.extend(cdata['users'])
#             data.update(cdata['user_data'])
        
#     clients = list(sorted(data.keys()))
#     return clients, data

# ## leaf data to dataloader
# def utils_loader(data, batch_size, train_or_test):
#     if args.dataset == 'mnist':
#         x = torch.reshape(torch.tensor(data['x']), (-1, 1, 28, 28))
#         y = torch.tensor(data['y'], dtype=torch.int64)
#     elif args.dataset == 'cifar10':
#         # x = torch.reshape(torch.tensor(data['x'], dtype=torch.float), (-1, 1, 32, 32))
#         x = torch.tensor(data['x'], dtype=torch.float)
#         # transform_train = transforms.Compose([
#         #     transforms.ToTensor(),
#         #     transforms.Normalize(**__imagenet_stats),
#         # ])
#         # x = transform_train(np.array(data['x']))
#         y = torch.tensor(data['y'], dtype=torch.int64)
#     elif args.dataset == 'celeba':
#         x = [load_image(i) for i in data['x']]
#         x = torch.reshape(torch.tensor(np.array(x), dtype=torch.float), (-1, 3, 84, 84))
#         x = F.normalize(x, p=2, dim=1)
#         # y = torch.tensor(np.array(data['y']), dtype=torch.int64)
#         y = torch.tensor(np.array(data['y']), dtype=torch.int64)
#     if train_or_test == 'train':
#         batch_size = batch_size
#     elif train_or_test == 'test':
#         batch_size = len(y)
#     torch_dataset = Data.TensorDataset(x, y)
#     data_leaf_loader = Data.DataLoader(
#         dataset=torch_dataset,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=2,
#         pin_memory=True
#     )
#     return data_leaf_loader

# def load_image(img_name):
#     IMAGES_DIR = '../fed_data/leaf/celeba/data/raw/img_align_celeba'
#     img = Image.open(os.path.join(IMAGES_DIR, img_name))
#     img = img.resize((84, 84)).convert('RGB')
#     return np.array(img)


# def partition_data(dataset, datadir, n_nets, alpha):

#     if dataset == 'cifar10':
#         X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
#         n_train = X_train.shape[0]

#         min_size = 0
#         K = 10
#         N = y_train.shape[0]
#         net_dataidx_map = {}

#         while min_size < 10:
#             idx_batch = [[] for _ in range(n_nets)]
#             # for each class in the dataset
#             for k in range(K):
#                 idx_k = np.where(y_train == k)[0]
#                 np.random.shuffle(idx_k)
#                 proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
#                 ## Balance
#                 proportions = np.array([p*(len(idx_j)<N/n_nets) for p,idx_j in zip(proportions,idx_batch)])
#                 proportions = proportions/proportions.sum()
#                 proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
#                 idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
#                 min_size = min([len(idx_j) for idx_j in idx_batch])

#         for j in range(n_nets):
#             np.random.shuffle(idx_batch[j])
#             net_dataidx_map[j] = idx_batch[j]

#         # N_test = y_test.shape[0]
#         # net_dataidx_map_test = {}

#     traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)

#     #return (X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts)
#     return y_train, net_dataidx_map, traindata_cls_counts


# def load_cifar10_data(datadir):

#     transform = transforms.Compose([transforms.ToTensor()])

#     cifar10_train_ds = CIFAR10_truncated(datadir, train=True, download=True, transform=transform)
#     cifar10_test_ds = CIFAR10_truncated(datadir, train=False, download=True, transform=transform)

#     X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
#     X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

#     return (X_train, y_train, X_test, y_test)


# def record_net_data_stats(y_train, net_dataidx_map):

#     net_cls_counts = {}

#     for net_i, dataidx in net_dataidx_map.items():
#         unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
#         tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
#         net_cls_counts[net_i] = tmp
#     # logging.debug('Data statistics: %s' % str(net_cls_counts))
#     return net_cls_counts


# def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None):

#     if dataset in ('mnist', 'cifar10'):
#         if dataset == 'mnist':
#             # dl_obj = MNIST_truncated

#             transform_train = transforms.Compose([
#                            transforms.ToTensor(),
#                            transforms.Normalize((0.1307,), (0.3081,))])

#             transform_test = transforms.Compose([
#                            transforms.ToTensor(),
#                            transforms.Normalize((0.1307,), (0.3081,))])

#         elif dataset == 'cifar10':
#             dl_obj = CIFAR10_truncated

#             normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
#                                 std=[x/255.0 for x in [63.0, 62.1, 66.7]])
#             transform_train = transforms.Compose([
#                 transforms.ToTensor(),
#                 transforms.Lambda(lambda x: F.pad(
#                                     Variable(x.unsqueeze(0), requires_grad=False),
#                                     (4,4,4,4),mode='reflect').data.squeeze()),
#                 transforms.ToPILImage(),
#                 transforms.RandomCrop(32),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ToTensor(),
#                 normalize,
#                 ])
#             # data prep for test set
#             transform_test = transforms.Compose([transforms.ToTensor(),normalize])


#         train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True)
#         test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True)

#         train_dl = Data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True)
#         test_dl = Data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False)

#     return train_dl, test_dl
