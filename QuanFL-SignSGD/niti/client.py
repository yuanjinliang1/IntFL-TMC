from types import new_class
import torch.nn as nn
import utils


from grad_compress.sign_sgd import SignSGDUpdate
class Client:

    def __init__(self, client_id, train_data, test_data, model=None, optimizer=None, model_type='int'):
        self.model = model
        self.id = client_id
        self.train_data = train_data
        self.test_data = test_data
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optimizer
        self.model_type = model_type

        self.compressor = SignSGDUpdate()

    @property
    def num_test_samples(self):
        """Number of test samples for this client.

        Return:
            int: Number of test samples for this client
        """
        if self.test_data is None:
            return 0
        return len(self.test_data.dataset.target)

    @property
    def num_train_samples(self):
        """Number of train samples for this client.

        Return:
            int: Number of train samples for this client
        """
        if self.train_data is None:
            return 0
        return len(self.train_data.dataset.target)
    
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
        # self.model.to('cuda:3')
        self.model.train()

        from meters import AverageMeter, accuracy
        losses = AverageMeter()
        top1 = AverageMeter()
        # top5 = AverageMeter()

        import copy
        # para = copy.deepcopy(self.model.state_dict())

        # data_loader = utils.utils_loader(self.train_data, batch_size, 'train')
        data_loader = self.train_data

        old_para = copy.deepcopy(self.model.state_dict())

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
        
        new_para = self.model.state_dict()

        # gradients = old_para-new_para

        layer_name_list_old, layer_val_list_old = list(old_para.keys()), list(old_para.values())  
        layer_name_list_new, layer_val_list_new = list(new_para.keys()), list(new_para.values())  

        gradients = []
        for i in range(len(layer_val_list_old)):
            gradients.append(layer_val_list_old[i] - layer_val_list_new[i])
        # for idx, l in enumerate(self.model.layers):
        #     if hasattr(l,'weight'):
        #         gradients.append(l.weight.grad)
        #         # writer.add_histogram('Grad/'+l.__class__.__name__ +'_'+str(idx), l.weight.grad, epoch*total_steps+i)
        # for idx, l in enumerate(self.model.classifier):
        #     if hasattr(l,'weight'):
        #         gradients.append(l.weight.grad)
        #         # writer.add_histogram('Grad/'+l.__class__.__name__ +'_'+str(idx), l.weight.grad, epoch*total_steps+i)
        grad, size_old, size_new = self.compressor.GradientCompress(gradients)
        update = self.model.state_dict()

        return losses.avg, update, grad


    def test(self, batch_size=5):
        # self.model.to('cuda:3')
        self.model.eval()

        from meters import AverageMeter, accuracy
        losses = AverageMeter()
        top1 = AverageMeter()

        import copy
        para = copy.deepcopy(self.model.state_dict())
        # data_loader = utils.utils_loader(self.test_data, batch_size, 'test')
        data_loader = self.test_data
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
            
            prec1, _ = accuracy(output.detach(), target, topk=(1, 2))
            top1.update(float(prec1), inputs.size(0))
        
        return top1


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