import logging
import numpy as np
# 导入 Python 的日志模块和 NumPy 库。
import torch
from torch import nn
# 导入 PyTorch 库和神经网络模块。


# 这段代码定义了一个名为 NormalTrainer 的类，它封装了训练和评估模型的标准流程。以下是对代码的逐行解释：
# 整体来看，NormalTrainer 类提供了一套完整的训练和评估流程，包括初始化、状态更新、参数获取和设置、梯度处理、学习率调度以及训练和测试。
# 它通过封装这些功能，使得模型的训练和评估过程更加标准化和模块化。

from utils.data_utils import (
    get_named_data,
    get_all_bn_params,
    check_device
)
# 从 utils.data_utils 导入数据相关的实用函数。
from trainers.averager import Averager
# 从 trainers.averager 导入 Averager 类。
from utils.set import *
from utils.log_info import *
from utils.tool import *
# 导入 utils.set、utils.log_info 和 utils.tool 模块中的所有内容。


class NormalTrainer(object):
    # 定义 NormalTrainer 类。
    def __init__(self, model, device, criterion, optimizer, lr_scheduler, args, **kwargs):
        # 构造函数，初始化训练器实例。
        if kwargs['role'] == 'server':
            if "server_index" in kwargs:
                self.server_index = kwargs["server_index"]
            else:
                self.server_index = args.server_index
            self.client_index = None
            self.index = self.server_index

        elif kwargs['role'] == 'client':
            if "client_index" in kwargs:
                self.client_index = kwargs["client_index"]
            else:
                self.client_index = args.client_index
            self.server_index = None
            self.index = self.client_index
        else:
            raise NotImplementedError

        self.role = kwargs['role']
        # 设置角色（服务器或客户端）。
        self.args = args
        self.model = model
        # 保存参数、模型、设备、标准、优化器和学习率调度器。
        self.device = device
        self.criterion = criterion.to(device)
        self.optimizer = optimizer
        # 将模型和标准移到设备上。

        self.param_groups = self.optimizer.param_groups  
        # 获取优化器的参数组。
        self.named_parameters = list(self.model.named_parameters())  # tuple [(name,param),(),...,()]
        # 获取模型的命名参数。
        if len(self.named_parameters) > 0:
            self._parameter_names = {v: k for k, v
                                    in sorted(self.named_parameters)}
            #print('Sorted named_parameters')
        else:
            self._parameter_names = {v: 'noname.%s' % i
                                    for param_group in self.param_groups
                                    for i, v in enumerate(param_group['params'])}

        self.averager = Averager(self.args, self.model) # it doesn't matter
        # 创建 Averager 实例。
        self.lr_scheduler = lr_scheduler

    def epoch_init(self):
        # 定义 epoch_init 方法，用于每个周期开始时的初始化操作。
        pass

    def epoch_end(self):
        # 定义 epoch_end 方法，用于每个周期结束时的操作。
        pass

    def track(self, tracker, summary_n_samples, model, loss, end_of_epoch,
            checkpoint_extra_name="centralized",
            things_to_track=[]):
        # 定义 track 方法，用于跟踪训练过程中的指标。
        pass

    def update_state(self, **kwargs):
        # 定义 update_state 方法，用于更新训练状态。
        # This should be called begin the training of each epoch.
        self.update_loss_state(**kwargs)

    def get_model_named_modules(self):
        # 定义 get_model_named_modules 方法，用于获取模型的命名模块。
        return dict(self.model.cpu().named_modules())

    def get_model(self):
        # 定义 get_model 方法，用于获取模型。
        return self.model

    def get_model_params(self):
        # 定义 get_model_params 方法，用于获取模型参数。
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        # 定义 set_model_params 方法，用于设置模型参数。
        # for name, param in model_parameters.items():
        #     logging.info(f"Getting params as model_parameters: name:{name}, shape: {param.shape}")
        self.model.load_state_dict(model_parameters)
        # 定义 set_feature_align_means 方法，用于设置特征对齐均值。

    def set_feature_align_means(self, feature_align_means):
        # 定义 get_feature_align_means 方法，用于获取特征对齐均值。
        self.feature_align_means = feature_align_means
        self.align_feature_loss.feature_align_means = feature_align_means

    def get_feature_align_means(self):
        return self.feature_align_means

    def get_model_bn(self):
        # 定义 get_model_bn 方法，用于获取模型的批量归一化参数。
        all_bn_params = get_all_bn_params(self.model)
        return all_bn_params

# got it Batch_Normalization set
    def set_model_bn(self, all_bn_params):
        # 定义 set_model_bn 方法，用于设置模型的批量归一化参数。
        # logging.info(f"all_bn_params.keys(): {all_bn_params.keys()}")
        # for name, params in all_bn_params.items():
            # logging.info(f"name:{name}, params.shape: {params.shape}")
        for module_name, module in self.model.named_modules():
            if type(module) is nn.BatchNorm2d:
                # logging.info(f"module_name:{module_name}, params.norm: {module.weight.data.norm()}")
                module.weight.data = all_bn_params[module_name+".weight"] 
                module.bias.data = all_bn_params[module_name+".bias"] 
                module.running_mean = all_bn_params[module_name+".running_mean"] 
                module.running_var = all_bn_params[module_name+".running_var"] 
                module.num_batches_tracked = all_bn_params[module_name+".num_batches_tracked"] 

#  `mode` choices: ['MODEL', 'GRAD', 'MODEL+GRAD']
    def get_model_grads(self):
        # 定义 get_model_grads 方法，用于获取模型梯度。
        named_mode_data = get_named_data(self.model, mode='GRAD', use_cuda=True)
        # logging.info(f"Getting grads as named_grads: {named_grads}")
        return named_mode_data

    def set_grad_params(self, named_grads):
        # 定义 set_grad_params 方法，用于设置梯度参数。
        # pass
        self.model.train()
        self.optimizer.zero_grad()
        for name, parameter in self.model.named_parameters():
            parameter.grad.copy_(named_grads[name].data.to(self.device)) # 把改name的grad复制过来

    def clear_grad_params(self):
        # 定义 clear_grad_params 方法，用于清除梯度参数。
        self.optimizer.zero_grad()

    def update_model_with_grad(self):
        # 定义 update_model_with_grad 方法，用于使用梯度更新模型。
        self.model.to(self.device)
        self.optimizer.step()

    def get_optim_state(self):
        # 定义 get_optim_state 方法，用于获取优化器状态。
        return self.optimizer.state

    def clear_optim_buffer(self):
        # 定义 clear_optim_buffer 方法，用于清除优化器缓冲区。
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.optimizer.state[p]
                # Reinitialize momentum buffer
                if 'momentum_buffer' in param_state:
                    param_state['momentum_buffer'].zero_()

    def lr_schedule(self, progress):
        # 定义 lr_schedule 方法，用于学习率调度。
        if self.lr_scheduler is not None:
            self.lr_scheduler.step(progress)
        else:
            logging.info("No lr scheduler...........")

    def warmup_lr_schedule(self, iterations):
        # 定义 warmup_lr_schedule 方法，用于预热学习率调度。
        if self.lr_scheduler is not None:
            self.lr_scheduler.warmup_step(iterations)

    def get_train_batch_data(self, train_dataloader):
        # 定义 get_train_batch_data 方法，用于获取训练批次数据。
        try:
            train_batch_data = self.train_local_iter.next()
            # logging.debug("len(train_batch_data[0]): {}".format(len(train_batch_data[0])))
            if len(train_batch_data[0]) < self.args.batch_size:
                logging.debug("WARNING: len(train_batch_data[0]): {} < self.args.batch_size: {}".format(
                    len(train_batch_data[0]), self.args.batch_size))
                # logging.debug("train_batch_data[0]: {}".format(train_batch_data[0]))
                # logging.debug("train_batch_data[0].shape: {}".format(train_batch_data[0].shape))
        except:
            self.train_local_iter = iter(train_dataloader)
            train_batch_data = self.train_local_iter.next()
        return train_batch_data

    def train_mix_dataloader(self, epoch, trainloader, device, **kwargs):
        # 定义 train_mix_dataloader 方法，用于使用混合数据加载器进行训练。
        self.model.to(device)
        self.model.train()
        self.model.training =True

        loss_avg = AverageMeter()
        acc = AverageMeter()

        logging.info('\n=> Training Epoch #%d, LR=%.4f' % (epoch, self.optimizer.param_groups[0]['lr']))
        for batch_idx, (x1, x2,x3, y1, y2,y3) in enumerate(trainloader):
            x1, x2, x3, y1, y2,y3 = x1.to(device), x2.to(device), x3.to(device), \
                                    y1.to(device), y2.to(device), y3.to(device)

            batch_size = x1.size(0)
            self.optimizer.zero_grad()

            x = torch.cat((x1, x2,x3))
            y = torch.cat((y1,y2,y3))

            out = self.model(x)

            loss = self.criterion(out, y)

            # ========================FedProx=====================#
            if self.args.fedprox:
                fed_prox_reg = 0.0
                previous_model = kwargs["previous_model"]
                for name, param in self.model.named_parameters():
                    fed_prox_reg += ((self.args.fedprox_mu / 2) * \
                        torch.norm((param - previous_model[name].data.to(device)))**2)
                loss += fed_prox_reg
            # ========================FedProx=====================#

            loss.backward()
            self.optimizer.step()

            # ========================SCAFFOLD=====================#
            if self.args.scaffold:
                c_model_global = kwargs['c_model_global']
                c_model_local = kwargs['c_model_local']
                if self.lr_scheduler is not None:
                    current_lr = self.lr_scheduler.lr
                else:
                    current_lr = self.args.lr
                for name, param in self.model.named_parameters():
                    # logging.debug(f"c_model_global[name].device : {c_model_global[name].device}, \
                    #     c_model_local[name].device : {c_model_local[name].device}")
                    param.data = param.data - 0.000001 * \
                                 check_device((c_model_global[name] - c_model_local[name]), param.data.device)
            # ========================SCAFFOLD=====================#

            prec1, prec5, correct, pred, _ = accuracy(out.data, y.data, topk=(1, 5))

            loss_avg.update(loss.data.item(), batch_size)
            acc.update(prec1.data.item(), batch_size)

            n_iter = (epoch - 1) * len(trainloader) + batch_idx

            log_info('scalar', '{role}_{index}_train_loss_epoch {epoch}'.format(role=self.role, index=self.index, epoch=epoch),
                     loss_avg.avg,step=n_iter,record_tool=self.args.record_tool, 
                        wandb_record=self.args.wandb_record)
            log_info('scalar', '{role}_{index}_train_acc_epoch {epoch}'.format(role=self.role, index=self.index, epoch=epoch),
                     acc.avg,step=n_iter,record_tool=self.args.record_tool,
                     wandb_record=self.args.wandb_record)

    def test_on_server_for_round(self, round, testloader, device):
        # 定义 test_on_server_for_round 方法，用于在服务器上进行测试。
        self.model.to(device)
        self.model.eval()

        test_acc_avg = AverageMeter()
        test_loss_avg = AverageMeter()

        total_loss_avg = 0
        total_acc_avg = 0
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(testloader):
                # distribute data to device
                x, y = x.to(device), y.to(device).view(-1, )
                batch_size = x.size(0)

                out = self.model(x)

                loss = self.criterion(out, y)
                prec1, _ = accuracy(out.data, y)

                n_iter = (round - 1) * len(testloader) + batch_idx
                test_acc_avg.update(prec1.data.item(), batch_size)
                test_loss_avg.update(loss.data.item(), batch_size)

                log_info('scalar', '{role}_{index}_test_acc_epoch'.format(role=self.role, index=self.index),
                         test_acc_avg.avg, step=n_iter,record_tool=self.args.record_tool,
                     wandb_record=self.args.wandb_record)
                total_loss_avg += test_loss_avg.avg
                total_acc_avg += test_acc_avg.avg
            total_acc_avg /= len(testloader)
            total_loss_avg /= len(testloader)
            log_info('scalar', '{role}_{index}_total_acc_epoch'.format(role=self.role, index=self.index),
                     total_acc_avg, step=round,record_tool=self.args.record_tool,
                     wandb_record=self.args.wandb_record)
            log_info('scalar', '{role}_{index}_total_loss_epoch'.format(role=self.role, index=self.index),
                     total_loss_avg, step=round, record_tool=self.args.record_tool,
                     wandb_record=self.args.wandb_record)
            return total_acc_avg
