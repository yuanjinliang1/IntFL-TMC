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
from log import setup_logging, ResultsLog, save_checkpoint
from meters import AverageMeter, accuracy
from preprocess import get_transform, get_int8_transform
from lenet import lenet
from vgg import VGG_cifar 
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
    # logging.info("ACC bitwidth: %d", ti_torch.ACC_BITWIDTH)

    # create clients
    # clients = setup_clients(model, train_loader)
    clients = setup_clients(args.model_type)

    # create server
    server = init_server(args.model_type)

    plot_result = []

    for epoch in range(args.start_epoch, args.num_round):
        # server select clients
        # if epoch==49 or epoch==99 or epoch==149 or epoch==199:
        # # if epoch==2 or epoch==3:
        #     server.select_clients(epoch, clients, num_clients=args.total_clients)
        # else:
        #     server.select_clients(epoch, clients, num_clients=args.num_clients)
        server.select_clients(epoch, clients, num_clients=args.num_clients)
        c_ids, c_num_samples = server.get_clients_info(server.selected_clients)
        # np.save('client_num_samples.npy', c_num_samples)
        # samples = np.load('client_num_samples.npy')
        error_clients = ['f2115_77', 'f3397_12', 'f2240_80', 'f0074_07', 'f1819_37', 'f0413_09', 'f1726_01', 
        'f1933_13', 'f0256_08', 'f1487_45', 'f0911_02', 'f1575_36', 'f1110_03', 'f3795_00', 'f2456_71', 'f1750_06']
        ## 从选中的参与训练的终端中剔除掉这些client
        # server.selected_clients
        # for client in server.selected_clients:
        #     if client.id in error_clients:
        #         selected_clients = list(server.selected_clients)
        #         selected_clients.remove(client)
        #         server.selected_clients = np.array(selected_clients)
        # train
        server.train_model(epoch, args.num_epochs, args.batch_size, server.selected_clients, args.model_type)

        ## 模型聚合之前，测试每个当前参与训练的终端在测试集上的精度表现
        # if epoch % args.log_interval == 0:
        #     ## server.test_model，在每个终端调用test时会取server存的模型，所以本轮训练的效果没有，建议去训练之后做这件事
        #     metrics = server.test_model(server.selected_clients, args.model_type)
        #     save_file_name = 'before-aggregate-float-mnist-' + str(args.num_clients) + '-round' + str(epoch) + '.npy'
        #     np.save(save_file_name, metrics)

        # aggregate
        server.update_model(args.model_type)
        
        # 模型聚合之后，测试全局模型在终端的精度
        if epoch % args.log_interval == 0:
            # clients = clients[0:2]
            ## 选用所有的终端测试 clients
            ## 选用本round参与训练的终端测试 selected_clients
            # metrics = server.test_model(clients, args.model_type)
            metrics = server.test_model(server.selected_clients, args.model_type)
            metrics_list = list(metrics.values())
            avg_acc = 0.
            total_weight = 0
            for metric in metrics_list:
                total_weight += metric.count
                avg_acc += metric.count * metric.avg
            avg_acc = avg_acc / total_weight
            plot_result.append((epoch, avg_acc))
            logging.info('current round {:d} '
                            'accuracy {:.3f} \n'
                            .format(epoch, avg_acc))
        if epoch % 100 == 0 and epoch != 0:
            save_file_name = 'result-niti-float-mnist-' + str(args.num_clients) + '-round' + str(epoch) + '-localEpoch' + str(args.num_epochs) +'.npy'
            np.save(save_file_name, plot_result)
    save_file_name = 'result-niti-float-mnist-' + str(args.num_clients) + '-round' + str(args.num_round) + '-localEpoch' + str(args.num_epochs) +'.npy'
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
            model= lenet().to('cuda:0')

        elif args.dataset == 'cifar10':
            if args.model == 'vgg':
                model= VGG_cifar(args.depth,10).to('cuda:0')

        num_parameters = sum([l.nelement() for l in model.parameters()])
        logging.info("created float network on %s", args.dataset)
        logging.info("number of parameters: %d", num_parameters)
        regime = getattr(model, 'regime')
        optimizer = OptimRegime(model.parameters(), regime)
    return model, optimizer


def setup_clients(model_type='int'):
    train_data_dir, test_data_dir = '../fed_data/leaf/femnist/data/train', '../fed_data/leaf/femnist/data/test'
    # users, train_data, test_data = read_data(train_data_dir, test_data_dir)
    users, train_data, test_data = read_data_test(train_data_dir, test_data_dir)
    if model_type == 'hybrid':
        nums = np.ones(len(users), int)
        half = int(len(users)/2)
        nums[:half] = 0
        np.random.shuffle(nums)
        clients = create_hybrid_clients(users, train_data, test_data, nums)
        print()
    else:
        model, optimizer = generate_model(model_type)
        clients = create_clients(users, train_data, test_data, model, optimizer, model_type)
    
    # users = ["00", "11"]
    # data = {"00": data, "11": data}
    # clients = create_clients(users, data, model)
    return clients

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

def read_data(train_data_dir, test_data_dir):
    train_clients, train_data = read_dir(train_data_dir)
    test_clients, test_data = read_dir(test_data_dir)
    return train_clients, train_data, test_data

def read_data_test(train_data_dir, test_data_dir):
    '''
    把不同终端的测试集集中到一起，组成新的global测试集，每个round参与训练的终端去这个测试集上验证性能
    '''
    train_clients, train_data = read_dir(train_data_dir)
    test_clients, test_data = read_dir(test_data_dir)
    va = list(test_data.values())
    temp = {'x': [], 'y': []}
    for element in va:
        temp = {key: temp[key] + element[key] for key in temp}
    for key in test_data.keys():
        test_data[key] = temp
    return train_clients, train_data, test_data

def read_dir(data_dir):
    clients = []
    from collections import defaultdict
    import os
    import json
    data = defaultdict(lambda : None)
    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        data.update(cdata['user_data'])

    clients = list(sorted(data.keys()))
    return clients, data


if __name__ == '__main__':
    main()
