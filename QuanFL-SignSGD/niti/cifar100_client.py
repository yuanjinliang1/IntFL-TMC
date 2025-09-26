import torch.nn as nn
class Cifar100Client:

    def __init__(self, client_id, data={'x' : [],'y' : []}, model=None, optimizer=None, model_type='int'):
        self.model = model
        self.id = client_id
        self.data = data
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optimizer
        self.model_type = model_type

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
        if self.data is None:
            return 0
        return len(self.data)
    
    def train(self, epoch, num_epochs=1, batch_size=5, model_type='int'):
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

        self.model.train()

        from meters import AverageMeter, accuracy
        losses = AverageMeter()
        top1 = AverageMeter()
        # top5 = AverageMeter()

        import copy
        para = copy.deepcopy(self.model.state_dict())

        data_loader = self.utils_loader(self.data, batch_size, 'train')
        for _ in range(num_epochs):
            for i, (inputs, target) in enumerate(data_loader):
                # measure data loading time
                # data_time.update(time.time() - end)
                inputs = inputs.to('cuda:3')
                target = target.to('cuda:3')

                # compute output
                output = self.model(inputs)
                self.criterion.to('cuda:3')
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


    def test(self, batch_size=5):
        self.model.eval()

        from meters import AverageMeter, accuracy
        losses = AverageMeter()
        top1 = AverageMeter()

        import copy
        para = copy.deepcopy(self.model.state_dict())
        data_loader = self.utils_loader(self.data, batch_size, 'test')
        for i, (inputs, target) in enumerate(data_loader):
            # measure data loading time
            # data_time.update(time.time() - end)
            inputs = inputs.to('cuda:3')
            target = target.to('cuda:3')

            # compute output
            output = self.model(inputs)
            if self.model_type == 'int':
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

    
    def utils_loader(self, data, batch_size, train_or_test):
        import torch
        data_x = []
        data_y = []
        for i in data:
            data_x.append(i['image'])
            data_y.append(i['label'])
        
        x = torch.tensor(data_x, dtype=torch.float32)
        x = x.permute(0,3,1,2)
        # x = torch.reshape(torch.tensor(data_x), (-1, 32, 3, 3, 3))
        y = torch.tensor(data_y, dtype=torch.int64)
        import torch.utils.data as Data
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


    def forward(self, data, model, criterion, epoch, training, optimizer=None):
        if training:
            model.train()
        else:
            model.eval()

        from meters import AverageMeter, accuracy

        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()


        # total_steps=len(data_loader)

        for i, (inputs, target) in enumerate(data):
            # measure data loading time
            # data_time.update(time.time() - end)
            inputs = inputs.to('cuda:3')
            target = target.to('cuda:3')

            # compute output
            output = model(inputs) # forward TiNet
            if model_type == 'int':
                # omit the output exponent
                output, output_exp = output
                output = output.float()
                loss = criterion(output*(2**output_exp.float()), target)
            else:
                output_exp = 0
                loss = criterion(output, target)

            # measure accuracy and record loss
            losses.update(float(loss), inputs.size(0))
            prec1, prec5 = accuracy(output.detach(), target, topk=(1, 5))
            top1.update(float(prec1), inputs.size(0))
            top5.update(float(prec5), inputs.size(0))

            if training:
                if model_type == 'int':
                    model.backward(target)

                elif model_type == 'hybrid':
                    # float backward
                    optimizer.update(epoch, epoch * len(data_loader) + i)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    #int8 backward
                    model.backward()
                else:
                    optimizer.update(epoch, epoch * len(data_loader) + i)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.log_interval == 0 and training:
                logging.info('{model_type} [{0}][{1}/{2}] '
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                            'Data {data_time.val:.2f} '
                            'loss {loss.val:.3f} ({loss.avg:.3f}) '
                            'e {output_exp:d} '
                            '@1 {top1.val:.3f} ({top1.avg:.3f}) '
                            '@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                                epoch, i, len(data_loader),
                                model_type=model_type,
                                batch_time=batch_time,
                                data_time=data_time,
                                loss=losses,
                                output_exp=output_exp,
                                top1=top1, top5=top5))

                if args.grad_hist:
                    if args.model_type == 'int':
                        for idx, l in enumerate(model.forward_layers):
                            if hasattr(l,'weight'):
                                grad = l.grad_int32acc
                                writer.add_histogram('Grad/'+l.__class__.__name__ +'_'+str(idx), grad, epoch*total_steps+i)

                    elif args.model_type == 'float':
                        for idx, l in enumerate(model.layers):
                            if hasattr(l,'weight'):
                                writer.add_histogram('Grad/'+l.__class__.__name__ +'_'+str(idx), l.weight.grad, epoch*total_steps+i)
                        for idx, l in enumerate(model.classifier):
                            if hasattr(l,'weight'):
                                writer.add_histogram('Grad/'+l.__class__.__name__ +'_'+str(idx), l.weight.grad, epoch*total_steps+i)

        return losses.avg, top1.avg, top5.avg