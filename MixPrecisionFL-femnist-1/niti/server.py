import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import numpy as np
import collections
import copy
import torch
import ti_torch
import logging
from optim import OptimRegime
from ti_lenet import TiLenet
from lenet import lenet
from args import parse_args
global args
args = parse_args()

class Server:
    def __init__(self, int_model, float_model):
        super().__init__()
        self.int_model = int_model
        self.float_model = float_model
        # self.model = client_model.parameters
        self.selected_clients = []
        self.int_updates = []
        self.float_updates = []

    def select_clients(self, my_round, possible_clients, num_clients=20):
        """Selects num_clients clients randomly from possible_clients.
        
        Note that within function, num_clients is set to
            min(num_clients, len(possible_clients)).

        Args:
            possible_clients: Clients from which the server can select.
            num_clients: Number of clients to select; default 20
        Return:
            list of (num_train_samples, num_test_samples)
        """
        for client in possible_clients:
            client.model_type = 'int'
        num_clients = min(num_clients, len(possible_clients))
        np.random.seed(my_round)
        self.selected_clients = np.random.choice(possible_clients, num_clients, replace=False)

        # return [(c.num_train_samples, c.num_test_samples) for c in self.selected_clients]

    def get_clients_info(self, clients):
        """Returns the ids, hierarchies and num_samples for the given clients.

        Returns info about self.selected_clients if clients=None;

        Args:
            clients: list of Client objects.
        """
        if clients is None:
            clients = self.selected_clients

        ids = [c.id for c in clients]
        num_samples = {c.id: c.num_train_samples for c in clients}
        return ids, num_samples

    def train_model(self, epoch, num_epochs=1, batch_size=5, clients=None, model_type='int'):
    # def train_model(self, num_epochs=1, batch_size=10, minibatch=None, clients=None, batch_num=1):
        """Trains self.model on given clients.
        
        Trains model on self.selected_clients if clients=None;
        each client's data is trained with the given number of epochs
        and batches.

        Args:
            clients: list of Client objects.
            num_epochs: Number of epochs to train.
            batch_size: Size of training batches.
            minibatch: fraction of client's data to apply minibatch sgd,
                None to use FedAvg
        Return:
            bytes_written: number of bytes written by each client to server 
                dictionary with client ids as keys and integer values.
            client computations: number of FLOPs computed by each client
                dictionary with client ids as keys and integer values.
            bytes_read: number of bytes read by each client from server
                dictionary with client ids as keys and integer values.
        """
        
        int_model_dict = None if self.int_model is None else copy.deepcopy(self.int_model.state_dict())
        float_model_dict = None if self.float_model is None else copy.deepcopy(self.float_model.state_dict())

        layer_name_list = list(int_model_dict.keys())
        float_values = list(float_model_dict.values())
        state_dict_agg = self.construct_dict(model_type='int', base=float_values, layer_name_list=layer_name_list)
        self.int_model.load_state_dict(state_dict_agg)
        int_model_dict = copy.deepcopy(self.int_model.state_dict())

        for c in clients:
            if c.model_type == 'int':
                model, optimizer = self.generate_model('int')
                c.model = model
                c.optimizer = optimizer
                regime = c.model.regime
                for s in regime:
                    if s['epoch'] == epoch:
                        ti_torch.GRAD_BITWIDTH = s['gb']
                        # logging.info('changing gradient bitwidth: %d', ti_torch.GRAD_BITWIDTH)
                        break
                c.model.load_state_dict(int_model_dict)
                loss, update = c.train(epoch, num_epochs, batch_size, c.model_type)
                self.int_updates.append((c.num_train_samples, copy.deepcopy(update)))

                c.model = None
                c.optimizer = None
            else:
                c.model.load_state_dict(float_model_dict)
                loss, update = c.train(epoch, num_epochs, batch_size, c.model_type)
                self.float_updates.append((c.num_train_samples, copy.deepcopy(update)))       
        # print("1111")


    def generate_model(self, model_type='int'):
        if model_type =='int':
            # logging.info('Create integer model')
            optimizer = None
            if args.dataset =='mnist':
                model = TiLenet()
            # elif args.dataset =='cifar10':
            #     if args.model == 'vgg':
            #         model = TiVGG_cifar(args.depth, 10).to('cuda:0')

            if args.weight_frac:
                regime = model.regime_frac
            else:
                regime = model.regime
        else:
            if args.dataset == 'mnist' and args.model == 'lenet':
                model= lenet().to('cuda:0')

            # elif args.dataset == 'cifar10':
            #     if args.model == 'vgg':
            #         model= VGG_cifar(args.depth,10).to('cuda:0')

            # num_parameters = sum([l.nelement() for l in model.parameters()])
            # logging.info("created float network on %s", args.dataset)
            # logging.info("number of parameters: %d", num_parameters)
            regime = getattr(model, 'regime')
            optimizer = OptimRegime(model.parameters(), regime)
        return model, optimizer

    def update_model(self, model_type='int'):
        if model_type == 'int':
            self.update_int_model()
        elif model_type == 'float':
            self.update_float_model()
        else:
            # self.update_hybrid_model()
            self.update_hybrid_model_3()
        # print()  
        # 

    def update_hybrid_model_3(self):
        layer_val_list_init_float, layer_name_list_init_float, _ = self.update_int_model_temp_3()

        state_dict_agg = self.construct_dict(model_type='float', base=layer_val_list_init_float, layer_name_list=layer_name_list_init_float)
        self.float_model.load_state_dict(state_dict_agg)
        # para = copy.deepcopy(self.int_model.state_dict())

        layer_name_list_int = list(self.int_model.state_dict().keys())
        state_dict_agg = self.construct_dict(model_type='int', base=layer_val_list_init_float, layer_name_list=layer_name_list_int)
        self.int_model.load_state_dict(state_dict_agg)
    

        # para_agg = self.int_model.state_dict()
        self.int_updates = [] 


    def update_int_model_temp_3(self):
        total_weight = 0.
        float_weight_list = []
        layer_name_list = []
        # result = sorted(self.int_updates,key=lambda x:(x[2]))
        # self.int_updates = result[0:5]
        layer_name_list_init_int, layer_val_list_init_int = list(self.int_model.state_dict().keys()), list(self.int_model.state_dict().values())            
        layer_beta_list = []
        for i in range(1, len(layer_val_list_init_int), 2):
            layer_beta_list.append(2**(layer_val_list_init_int[i].float())/2)
        for (client_samples, client_model) in self.int_updates:
            weight = []
            weight_exp = []
            float_weight = []
            total_weight += client_samples
            layer_name_list, layer_val_list = list(client_model.keys()), list(client_model.values())
            
            for i in range(0, len(layer_val_list), 2):
                layer_int = (layer_val_list[i], layer_val_list[i+1])                    
                layer_float = ti_torch.TiInt8ToFloat(layer_int)

                layer_int_init = (layer_val_list_init_int[i], layer_val_list_init_int[i+1])                    
                layer_float_init = ti_torch.TiInt8ToFloat(layer_int_init)

                layer_float = layer_float_init - layer_float 

                float_weight.append(float(client_samples) * layer_float)
            float_weight_list.append(float_weight)
            
        base = [0] * len(float_weight_list[0])
        for i in range(len(float_weight_list)):
            for j, v in enumerate(float_weight_list[i]):
                base[j] += v / total_weight
        
        layer_name_list_init_float, layer_val_list_init_float = list(copy.deepcopy(self.float_model.state_dict()).keys()), list(copy.deepcopy(self.float_model.state_dict()).values())            
        # base_init = []
        # for i in range(len(layer_val_list_init)):
        #     layer_int_init = (layer_val_list_init[i], layer_val_list_init[i+1])                    
        #     layer_float_init = ti_torch.TiInt8ToFloat(layer_int_init)
        #     base_init.append(layer_float_init)


        # for i in range(len(base)):
        #     shape = base[i].shape
        #     temp = base[i].reshape(-1)
        #     for j in range(len(temp)):
        #         if abs(temp[j]) <= layer_beta_list[i]:
        #             if temp[j] >= 0:
        #                 temp[j] += layer_beta_list[i]
        #             else:
        #                 temp[j] -= layer_beta_list[i]
        #     compensate_updates = temp.reshape(shape)
        #     layer_val_list_init_float[i] = layer_val_list_init_float[i] - compensate_updates
        
        for i in range(len(base)):
            layer_val_list_init_float[i] = layer_val_list_init_float[i] - base[i]
        # layer_val_list_init_float[i] = layer_val_list_init_float[i] - base[i]
        
        return layer_val_list_init_float, layer_name_list_init_float, total_weight


    def update_int_model(self):
        """
        """
        base, layer_name_list, _ = self.update_int_model_temp()
        # quant in construct_dict
        state_dict_agg = self.construct_dict(model_type='int', base=base, layer_name_list=layer_name_list)
        para = copy.deepcopy(self.int_model.state_dict())
        self.int_model.load_state_dict(state_dict_agg)
        para_agg = self.int_model.state_dict()
        self.int_updates = []
        # return total_weight, layer_name_list
    

    def update_int_model_temp(self):
        total_weight = 0.
        float_weight_list = []
        layer_name_list = []
        for (client_samples, client_model) in self.int_updates:
            weight = []
            weight_exp = []
            float_weight = []
            total_weight += client_samples
            layer_name_list, layer_val_list = list(client_model.keys()), list(client_model.values())
            for i in range(0, len(layer_val_list), 2):
                layer_int = (layer_val_list[i], layer_val_list[i+1])                    
                layer_float = ti_torch.TiInt8ToFloat(layer_int)
                float_weight.append(float(client_samples) * layer_float)
            float_weight_list.append(float_weight)
            
        base = [0] * len(float_weight_list[0])
        for i in range(len(float_weight_list)):
            for j, v in enumerate(float_weight_list[i]):
                base[j] += v / total_weight
        
        return base, layer_name_list, total_weight

    
    def update_float_model(self):
        total_weight = 0.
        base = [0] * len(self.float_updates[0][1])
        layer_name_list = []
        for (client_samples, client_model) in self.float_updates:
            layer_name_list = list(client_model.keys())
            total_weight += client_samples
            for i, v in enumerate(list(client_model.values())):
                base[i] += (client_samples * v.float())
        
        avg_base = [v/total_weight for v in base]

        state_dict_agg = self.construct_dict(model_type='float', base=avg_base, layer_name_list=layer_name_list)
        para = copy.deepcopy(self.float_model.state_dict())
        self.float_model.load_state_dict(state_dict_agg)
        self.float_updates = []
        return total_weight, layer_name_list


    def update_hybrid_model(self):
        # update int_updates and float_updates
        total_float_weight, float_layer_name = self.update_float_model()
        int_base, int_layer_name, total_int_weight = self.update_int_model_temp()
        float_model_dict = list(self.float_model.state_dict().values())
        float_base = float_model_dict[::2]
        # total_int_weight = update_int_model()
        # total_float_weight, layer_name_list= update_float_model()

        # for 
        # change int model to float model
        total_weight = total_int_weight + total_float_weight
        base = [0] * len(int_base)
        # update two new float model
        for i in range(len(int_base)):
            base[i] = (total_float_weight * float_base[i].float() + total_int_weight * int_base[i].float()) / total_weight

        int_model_avg = base
        state_dict_agg_int = self.construct_dict(model_type='int', base=int_model_avg, layer_name_list=int_layer_name)
        para = copy.deepcopy(self.int_model.state_dict())
        self.int_model.load_state_dict(state_dict_agg_int)
        para_agg = self.int_model.state_dict()
        self.int_updates = []
        
        float_model_avg = [0] * len(float_model_dict)
        for i in range(len(float_model_dict)):
            if i % 2 == 0:
                float_model_avg[i] = base[int(i/2)]
            else:
                float_model_avg[i] = float_model_dict[i]
        state_dict_agg_float = self.construct_dict(model_type='float', base=float_model_avg, layer_name_list=float_layer_name)
        para = copy.deepcopy(self.float_model.state_dict())
        self.float_model.load_state_dict(state_dict_agg_float)
        self.float_updates = []


    def construct_dict(self, model_type, base, layer_name_list):
        state_dict_agg=collections.OrderedDict()
        if model_type == 'int':
            layer_index = 0
            for float_weight in base:
                # quant float tensor to int tensor
                round_val, act_exp = ti_torch.weight_quant(float_weight)
                state_dict_agg[layer_name_list[layer_index]] = round_val
                state_dict_agg[layer_name_list[layer_index+1]] = act_exp
                layer_index += 2
        else:
            layer_index = 0
            for weight in base:
                state_dict_agg[layer_name_list[layer_index]] = weight
                layer_index += 1

        return state_dict_agg   



    def test_model(self, clients_to_test, model_type='int'):
        """
        """
        metrics = {}
        for c in clients_to_test:
            if c.model_type == 'int':
                model, _ = self.generate_model('int')
                c.model = model
                c.model.load_state_dict(self.int_model.state_dict())
            elif c.model_type == 'float':
                c.model.load_state_dict(self.float_model.state_dict())

            top1 = c.test(c.model_type)
            # acc_samples = (top1, c.num_test_samples)
            metrics[c.id] = top1
        return metrics


GPU='cuda:0'

BITWIDTH = 7