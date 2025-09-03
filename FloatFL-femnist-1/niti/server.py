import numpy as np
import collections
import copy
import torch
import ti_torch
import logging
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
        self.select_clients_acc_loss = []
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

        # if epoch % 10 == 0 and epoch != 0:
        #     print('save id, samples, acc loss')
        for c in clients:
            if c.model_type == 'int':
                regime = c.model.regime
                for s in regime:
                    if s['epoch'] == epoch:
                        ti_torch.GRAD_BITWIDTH = s['gb']
                        logging.info('changing gradient bitwidth: %d', ti_torch.GRAD_BITWIDTH)
                        break
                c.model.load_state_dict(int_model_dict)
                loss, update = c.train(epoch, num_epochs, batch_size, c.model_type)
                self.int_updates.append((c.num_train_samples, copy.deepcopy(update)))
            else:
                c.model.load_state_dict(float_model_dict)
                loss, update = c.train(epoch, num_epochs, batch_size, c.model_type)
                self.float_updates.append((c.num_train_samples, copy.deepcopy(update)))
            # if epoch % 50 == 0:
            # if epoch==49 or epoch==99 or epoch==149 or epoch==199:
            # # if epoch==2 or epoch==3:
            #     acc = c.test()
            #     self.select_clients_acc_loss.append({"client_id": c.id, "num_train_samples": c.num_train_samples, 
            #         "acc": acc.avg, "loss": loss})
            # print(c.id + " ------- train finish")
        print("------- all selected clients train finish")
        # if epoch % 10 ==0:
        # if epoch==49 or epoch==99 or epoch==149 or epoch==199:
        # # if epoch==2 or epoch==3:
        #     save_file_name = "./id_samples_acc_loss/id_samples_acc_loss-clientNum" + str(args.total_clients) + "-round" + str(epoch) + "-localEpoch" + str(args.num_epochs) + ".npy"
        #     np.save(save_file_name, self.select_clients_acc_loss)


    def update_model(self, model_type='int'):
        if model_type == 'int':
            self.update_int_model()
        elif model_type == 'float':
            self.update_float_model()
        else:
            self.update_hybrid_model()
        print()   


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
                c.model.load_state_dict(self.int_model.state_dict())
            elif c.model_type == 'float':
                c.model.load_state_dict(self.float_model.state_dict())

            top1 = c.test()
            # acc_samples = (top1, c.num_test_samples)
            metrics[c.id] = top1
        return metrics


GPU='cuda:0'

BITWIDTH = 7