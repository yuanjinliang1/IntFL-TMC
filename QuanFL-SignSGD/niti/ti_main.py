#!/usr/bin/python3
import os
import sys
import logging
import argparse
import numpy as np
import time
import torch
import torch.nn as nn
from datetime import datetime
from torchvision import datasets, transforms
# import tensorflow_federated as tff
# import tensorflow_datasets as tfds
from log import setup_logging, ResultsLog, save_checkpoint
from meters import AverageMeter, accuracy
from preprocess import get_transform, get_int8_transform
from lenet import lenet
from lenet import lenet_celeba
from mobilenet import mobilenet
import utils
# import cifar_data_loader

from vgg import VGG_cifar 
from vgg import VGG_celeba
from optim import OptimRegime
from data import get_dataset
from ti_lenet import TiLenet
from ti_vgg import TiVGG_cifar
import torch.optim as optim
import torch.nn.functional as F
import ti_torch
from torch.utils.tensorboard import SummaryWriter
import shutil

from client import Client
from server import Server
from args import parse_args

import data_loader

def main():
    #check gpu exist
    if not torch.cuda.is_available():
        print('Require nvidia gpu with tensor core to run')
        return

    global args
    args = parse_args()

    # random seed config
    if args.seed > 0:
        torch.manual_seed(args.seed)
        logging.info("random seed: %s", args.seed)
    else:
        logging.info("random seed: None")

    logging.info("act rounding scheme: %s", ti_torch.ACT_ROUND_METHOD.__name__)
    logging.info("err rounding scheme: %s", ti_torch.ERROR_ROUND_METHOD.__name__)
    logging.info("gradient rounding scheme: %s", ti_torch.GRAD_ROUND_METHOD.__name__)
    if args.weight_frac:
        ti_torch.UPDATE_WITH_FRAC = True
        logging.info("Update WITH Fraction")
    else:
        ti_torch.UPDATE_WITH_FRAC = False

    if args.weight_decay: 
        ti_torch.WEIGHT_DECAY = True
        logging.info("Update WITH WEIGHT DECAY")
    else:
        ti_torch.WEIGHT_DECAY = False
    
    y_train, net_dataidx_map, traindata_cls_counts = utils.partition_data(args.dataset, './cifar10', 100, 0.5)
    # DEFAULT_TRAIN_CLIENTS_NUM, train_data_num, test_data_num, train_data_global, test_data_global, \
    #     data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num = data_loader.load_partition_data_federated_cifar100('cifar100', 'fed_cifar100/datasets')

    import numpy as np
    # np.save('net_dataidx_map.npy', net_dataidx_map)
    # net_dataidx_map = np.load('net_dataidx_map.npy', allow_pickle=True).item()
    local_datasets = []
    for client_id in range(100):
        dataidxs = net_dataidx_map[client_id]
        train_dataloader, test_dataloader = utils.get_dataloader(args.dataset, './cifar10', 64, 64, dataidxs)
        local_datasets.append((train_dataloader, test_dataloader))
    
    clients = setup_clients(args.model_type, train_data_local_dict, test_data_local_dict)
    # clients = setup_clients(args.model_type, local_datasets)
    # clients = setup_clients(args.model_type, args.dataset)
    # clients = clients[0:10]

    # create server
    server = init_server(args.model_type)

    plot_result = []

    for epoch in range(args.start_epoch, args.num_round):
        server.select_clients(epoch, clients, num_clients=args.num_clients)
        # if epoch==199 or epoch==299 or epoch==399:
        # # if epoch==2 or epoch==3:
        #     server.select_clients(epoch, clients, num_clients=args.total_clients)
        # else:
        #     server.select_clients(epoch, clients, num_clients=args.num_clients)
        # c_ids, c_num_samples = server.get_clients_info(server.selected_clients)
        # import numpy as np
        # np.save('client_num_samples.npy', c_num_samples)
        # train
        server.train_model(epoch, args.num_epochs, args.batch_size, server.selected_clients, args.model_type)
        # aggregate
        # server.update_model(args.model_type)

        server.update_using_compressed_grad()

        if epoch % args.log_interval == 0:
            metrics = server.test_model(clients, args.model_type)
            metrics_list = list(metrics.values())
            avg_acc = 0.
            total_weight = 0
            ## cifar10 float 测试用的batch size = 10000，所以val=avg，没有影响，不过最好改了
            for metric in metrics_list:
                total_weight += metric.count
                avg_acc += metric.count * metric.avg
            avg_acc = avg_acc / total_weight
            plot_result.append((epoch, avg_acc))
            logging.info('current round {:d} '
                            'accuracy {:.3f} \n'
                            .format(epoch, avg_acc))
            # print(avg_acc)
            # print("epoch: ", epoch)
    
        if epoch % 100 == 0:
            save_file_name = 'result-niti-float-cifar10-' + str(args.num_clients) + '-round' + str(epoch) + '-localEpoch' + str(args.num_epochs) +'-epoch.npy'
            np.save(save_file_name, plot_result)
    save_file_name = 'result-niti-float-cifar10-' + str(args.num_clients) + '-round' + str(args.num_round) + '-localEpoch' + str(args.num_epochs) +'-epoch.npy'
    np.save(save_file_name, plot_result)


def init_server(model_type='int'):
    if model_type == 'int':
        int_model, _ = generate_model(model_type)
        server = Server(int_model, None)
    elif model_type == 'float':
        float_model, _ = generate_model(model_type)
        server = Server(None, float_model)
    else:
        int_model, _ = generate_model('int')
        float_model, _ = generate_model('float')
        server = Server(int_model, float_model)
    return server


def generate_model(model_type='int'):
    if model_type =='int':
        logging.info('Create integer model')
        optimizer = None
        if args.dataset =='mnist':
            model = TiLenet()
        elif args.dataset =='cifar10':
            if args.model == 'vgg':
                model = TiVGG_cifar(args.depth, 10)

        if args.weight_frac:
            regime = model.regime_frac
        else:
            regime = model.regime
    else:
        if args.dataset == 'mnist' and args.model == 'lenet':
            model= lenet()
        
        elif args.dataset == 'celeba':
            if args.model == 'lenet':
                model= lenet_celeba()
            elif args.model == 'vgg':
                model = VGG_celeba(84*84, 2)

        elif args.dataset == 'cifar10':
            if args.model == 'vgg':
                model= VGG_cifar(args.depth,10).to('cuda:3')
        elif args.dataset == 'cifar100':
            if args.model == 'vgg':
                model= VGG_cifar(args.depth,100).to('cuda:3')
        num_parameters = sum([l.nelement() for l in model.parameters()])
        logging.info("created float network on %s", args.dataset)
        logging.info("number of parameters: %d", num_parameters)
        regime = getattr(model, 'regime')
        optimizer = OptimRegime(model.parameters(), regime)
    return model, optimizer


# def setup_clients(model_type='int', dataset='mnist'):
#     if dataset == 'mnist':
#         train_data_dir, test_data_dir = '../fed_data/leaf/femnist/data/train', '../fed_data/leaf/femnist/data/test'
#         users, train_data, test_data = utils.read_data(train_data_dir, test_data_dir)
#     elif dataset == 'celeba':
#         train_data_dir, test_data_dir = '../fed_data/leaf/celeba/data/train', '../fed_data/leaf/celeba/data/test'
#         users, train_data, test_data = utils.read_data(train_data_dir, test_data_dir)
#     elif dataset == 'cifar10':
#         train_data_dir, test_data_dir = '../fed_data/cifar10/train', '../fed_data/cifar10/test'
    
#     if model_type == 'hybrid':
#         nums = np.ones(len(users), int)
#         half = int(len(users)/2)
#         nums[:half] = 0
#         np.random.shuffle(nums)
#         clients = create_hybrid_clients(users, train_data, test_data, nums)
#         print()
#     else:
#         model, optimizer = generate_model(model_type)
#         clients = create_clients(users, train_data, test_data, model, optimizer, model_type)
    
#     return clients

def setup_clients(model_type='int', train_data_local_dict=None, test_data_local_dict=None):
    # users = [str(i) for i in range(16)]
    clients = []
    model, optimizer = generate_model(model_type)
    users = list(train_data_local_dict.keys())
    train_data_loader = list(train_data_local_dict.values())
    test_data_loader = list(test_data_local_dict.values())
    for i in range(len(users)):
        u = users[i]
        train_data = train_data_loader[i]
        test_data = test_data_loader[i]
        client = Client(u, train_data, test_data, model, optimizer, model_type)
        clients.append(client)
    return clients

# def setup_clients(model_type='int', dataset=None):
#     # users = [str(i) for i in range(16)]
#     clients = []
#     model, optimizer = generate_model(model_type)
#     for i in range(len(dataset)):
#         u = str(i)
#         train_data = dataset[i][0]
#         test_data = dataset[i][1]
#         client = Client(u, train_data, test_data, model, optimizer, model_type)
#         clients.append(client)
#     return clients
    

def create_clients(users, train_data, test_data, model, optimizer, model_type):
    clients = [Client(u, train_data[u], test_data[u], model, optimizer, model_type) for u in users]
    return clients

def create_hybrid_clients(users, train_data, test_data, model_types):
    clients = []
    model_type_str = ''
    for i, model_type in enumerate(model_types):
        if model_type == 1:
            model_type_str = 'int'
        else:
            model_type_str = 'float'
        model, optimizer = generate_model(model_type_str)
        clients.append(Client(users[i], train_data[users[i]], test_data[users[i]], model, optimizer, model_type_str))
    return clients

# def read_data(train_data_dir, test_data_dir):
#     train_clients, train_data = read_dir(train_data_dir)
#     test_clients, test_data = read_dir(test_data_dir)
#     return train_clients, train_data, test_data

# def read_dir(data_dir):
#     clients = []
#     from collections import defaultdict
#     import os
#     import json
#     data = defaultdict(lambda : None)
#     files = os.listdir(data_dir)
#     files = [f for f in files if f.endswith('.json')]
#     for f in files:
#         file_path = os.path.join(data_dir,f)
#         with open(file_path, 'r') as inf:
#             cdata = json.load(inf)
#         clients.extend(cdata['users'])
#         data.update(cdata['user_data'])

#     clients = list(sorted(data.keys()))
#     return clients, data


if __name__ == '__main__':
    main()
