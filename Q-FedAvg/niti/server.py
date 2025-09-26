import numpy as np
import collections
import copy
from args import parse_args
global args
args = parse_args()
class Server:
    def __init__(self, model):
        super().__init__()
        self.model = model
        # self.model = client_model.parameters
        self.selected_clients = []
        self.select_clients_acc_loss = []
        self.updates = []

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

    def train_model(self, epoch, num_epochs=1, batch_size=10, clients=None, model_type='int'):
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
        import copy
        model_dict = copy.deepcopy(self.model.state_dict())
        for c in clients:
            c.model.load_state_dict(model_dict)
            loss, update = c.train(epoch, num_epochs, batch_size, model_type)
            self.updates.append((c.num_train_samples, copy.deepcopy(update)))
            # if epoch % 10 == 0:
            #     acc = c.test()
            #     # self.select_clients_acc_loss.append({"client_id": c.id, "num_train_samples": c.num_train_samples, 
            #     #     "acc": acc.avg, "loss": loss})
            print(c.id + " ------- train finish")
        print("------- all selected clients train finish")
        # if epoch % 10 ==0:
        #     save_file_name = "./id_samples_acc_loss/id_samples_acc_loss-clientNum" + str(args.num_clients) + "-round" + str(epoch) + "-localEpoch" + str(args.num_epochs) + ".npy"
        #     np.save(save_file_name, self.select_clients_acc_loss)

    def update_model(self):
        """
        """
        total_weight = 0.
        float_weight_list = []
        client_samples_list = []
        layer_name_list = []
        for (client_samples, client_model) in self.updates:
            weight = []
            weight_exp = []
            float_weight = []
            total_weight += client_samples
            client_samples_list.append(client_samples)
            layer_name_list = list(client_model.keys())
            state_dict_weight=collections.OrderedDict()
            state_dict_weight_exp=collections.OrderedDict()
            for key in client_model:
                key_list = key.split('.')
                if key_list[-1] == 'weight':
                    weight_temp = (key, client_model[key].float())
                    # state_dict_weight[key] = client_model[key].float()
                    weight.append(weight_temp)
                elif key_list[-1] == 'weight_exp':
                    weight_exp_temp = (key, client_model[key])
                    # state_dict_weight_exp[key] = client_model[key]
                    weight_exp.append(weight_exp_temp)
            for i in range(len(weight)):
                float_weight.append(weight[i][1] * (2**(weight_exp[i][1].float())))
            float_weight_list.append(float_weight)
            

        base = [0] * len(float_weight_list[0])
        for i in range(len(client_samples_list)):
            for j, v in enumerate(float_weight_list[i]):
                base[j] += float(client_samples_list[i]) / total_weight * v

        state_dict_agg=collections.OrderedDict()
        layer_index = 0
        for float_weight in base:
            import torch
            float_weight_range = torch.max(torch.abs(float_weight))
            float_weight_bitwidth=torch.ceil(torch.log2(float_weight_range))
            act_exp = float_weight_bitwidth - BITWIDTH
            temp = torch.round(float_weight/float_weight_range*(2**BITWIDTH-1))
            round_val = torch.round(float_weight/float_weight_range*(2**BITWIDTH-1)).type(torch.int8).to(GPU)
            # return quantised int8 and exponent
            act_exp = act_exp.type(torch.int8).to(GPU)
            state_dict_agg[layer_name_list[layer_index]] = round_val
            state_dict_agg[layer_name_list[layer_index+1]] = act_exp
            layer_index += 2
        para = copy.deepcopy(self.model.state_dict())
        self.model.load_state_dict(state_dict_agg)
        para_agg = self.model.state_dict()
        self.updates = []

    
    def update_float_model(self):
        total_weight = 0.
        base = [0] * len(self.updates[0][1])
        layer_name_list = []
        for (client_samples, client_model) in self.updates:
            layer_name_list = list(client_model.keys())
            total_weight += client_samples
            for i, v in enumerate(list(client_model.values())):
                base[i] += (client_samples * v.float())
        
        avg_base = [v/total_weight for v in base]
        state_dict_agg=collections.OrderedDict()
        layer_index = 0
        for weight in avg_base:
            state_dict_agg[layer_name_list[layer_index]] = weight
            layer_index += 1
        para = copy.deepcopy(self.model.state_dict())
        self.model.load_state_dict(state_dict_agg)
        self.updates = []


    def test_model(self, clients_to_test, model_type='int'):
        """
        """
        metrics = {}
        for c in clients_to_test:
            c.model.load_state_dict(self.model.state_dict())
            top1 = c.test(model_type)
            # acc_samples = (top1, c.num_test_samples)
            metrics[c.id] = top1
        return metrics


GPU='cuda:0'

BITWIDTH = 7