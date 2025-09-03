import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
class Client:

    def __init__(self, client_id, train_data={'x' : [],'y' : []}, test_data={'x' : [],'y' : []}, model=None, optimizer=None):
        self.model = model
        self.id = client_id
        self.train_data = train_data
        self.test_data = test_data
        import torch.nn as nn
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optimizer
    
    @property
    def num_test_samples(self):
        """Number of test samples for this client.

        Return:
            int: Number of test samples for this client
        """
        if self.test_data is None:
            return 0
        return len(self.test_data['y'])

    @property
    def num_train_samples(self):
        """Number of train samples for this client.

        Return:
            int: Number of train samples for this client
        """
        if self.train_data is None:
            return 0
        return len(self.train_data['y'])
    
    def train(self, epoch, num_epochs=1, batch_size=10, model_type='int'):
        """Trains on self.model using the client's train_data.

        Args:
            num_epochs: Number of epochs to train. Unsupported if minibatch is provided (minibatch has only 1 epoch)
            batch_size: Size of training batches.
            minibatch: fraction of client's data to apply minibatch sgd,
                None to use FedAvg
        Return:
            comp: number of FLOPs executed in training process
            num_samples: number of samples used in training
            update: set of weights
            update_size: number of bytes in update
        """
        # train_loss, train_prec1, train_prec5= self.forward(
        #     self.train_data, self.model, self.criterion, num_epochs, training=True, optimizer = self.optimizer)
        # if training:
        #     model.train()
        # else:
        #     model.eval()
        self.model.to('cuda:0')
        self.model.train()

        from meters import AverageMeter, accuracy
        losses = AverageMeter()
        top1 = AverageMeter()
        # top5 = AverageMeter()

        import copy
        para = copy.deepcopy(self.model.state_dict())
        
        ## construct data loader, train batch size = 5
        data_loader = self.utils_loader(self.train_data, 'train')
        for _ in range(num_epochs):
            for i, (inputs, target) in enumerate(data_loader):
                if i*5 > len(data_loader.dataset)/2:
                    break
                # measure data loading time
                # data_time.update(time.time() - end)
                inputs = inputs.to('cuda:0')
                target = target.to('cuda:0')

                # compute output
                output = self.model(inputs)
                self.criterion.to('cuda:0')
                if model_type == 'int':
                    # omit the output exponent
                    output, output_exp = output
                    output = output.float()
                    loss = self.criterion(output*(2**output_exp.float()), target)
                else:
                    output_exp = 0
                    loss = self.criterion(output, target)

                # measure accuracy and record loss
                losses.update(float(loss), inputs.size(0))
                # prec1, prec5 = accuracy(output.detach(), target, topk=(1, 5))
                # top1.update(float(prec1), inputs.size(0))
                # top5.update(float(prec5), inputs.size(0))

                if model_type == 'int':
                    self.model.backward(target)
                else:
                    self.optimizer.update(epoch, epoch * len(data_loader) + i)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

        update = self.model.state_dict()

        return losses.avg, update


    def test(self, model_type='int'):
        self.model.eval()

        from meters import AverageMeter, accuracy
        losses = AverageMeter()
        top1 = AverageMeter()

        import copy
        para = copy.deepcopy(self.model.state_dict())
        ## construct data loader, train batch size = full test data size
        data_loader = self.utils_loader(self.test_data, 'test')
        for i, (inputs, target) in enumerate(data_loader):
            # measure data loading time
            # data_time.update(time.time() - end)
            inputs = inputs.to('cuda:0')
            target = target.to('cuda:0')

            # compute output
            output = self.model(inputs)
            if model_type == 'int':
                # omit the output exponent
                output, output_exp = output
                output = output.float()
                loss = self.criterion(output*(2**output_exp.float()), target)
            else:
                output_exp = 0
                loss = self.criterion(output, target)

            # measure accuracy and record loss
            losses.update(float(loss), inputs.size(0))
            prec1, prec5 = accuracy(output.detach(), target, topk=(1, 5))
            top1.update(float(prec1), inputs.size(0))
        
        return top1

    
    def utils_loader(self, data, train_or_test):
        import torch
        x = torch.reshape(torch.tensor(data['x']), (-1, 1, 28, 28))
        y = torch.tensor(data['y'], dtype=torch.int64)
        import torch.utils.data as Data
        if train_or_test == 'train':
            batch_size = 5
        elif train_or_test == 'test':
            # batch_size = 5
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