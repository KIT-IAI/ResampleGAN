from torch import nn
import torch
import math
from ResampleGAN.TransGAN.LossFunction.LossFunction import loss_registry


class CombinedLoss(nn.Module):
    """
    组合多个损失函数

    :param weights: 损失权重字典，键为损失函数名称，值为长度为 max_epoch 的列表，每个元素表示对应 epoch 的权重
    :param use_losses: 要使用的损失函数列表，如果为None则使用所有注册的损失函数
    :param max_epoch: 最大训练轮数，用于验证权重列表的长度
    :param with_true: 是否使用 `LossFunctionTrue` 中的注册损失函数
    """

    def __init__(self, weights=None, use_losses=None, max_epoch=100, with_true=False, learnable=False):
        super().__init__()

        self.registry = loss_registry
        self.max_epoch = max_epoch
        self.learnable = learnable

        # 获取默认权重
        self.weights = weights

        # 初始化要使用的损失函数
        self.use_losses = use_losses or list(self.weights.keys())

        if self.learnable:
            self.log_vars = nn.ParameterDict()
            for name in self.use_losses:
                if name in self.weights:
                    init_val = -math.log(self.weights[name])  # 因为权重 = exp(-log_var)
                else:
                    init_val = 0.0
                self.log_vars[name] = nn.Parameter(torch.tensor([init_val], dtype=torch.float32))
            # 默认每个 loss 启动，使用 log_vars 控制权重
            self.weights = {name: [1.0]*self.max_epoch for name in self.use_losses}
        else:
            self.log_vars = None

        
        # 创建损失函数实例
        self.loss_functions = {
            name: self.registry.get_loss(name)()
            for name in self.use_losses
            if len(self.weights.get(name, [])) > 0 and any(w > 0 for w in self.weights[name])
        }

    def calculate_dynamic_weight(self, name, epoch):
        """
        根据当前epoch返回对应权重

        :param name: 损失函数名称
        :param epoch: 当前epoch
        :return: 当前 epoch 的权重
        """
        if name == 'feature_space':
            return self.weights[name][epoch] * min(epoch / self.max_epoch, 1.0)
        return self.weights[name][epoch]


    def forward(self, y_pred, y_ori, epoch, **kwargs):
        """
        计算总损失

        :param y_pred: 预测值
        :param y_ori: 真实值
        :param epoch: 当前训练轮数
        :return: total_loss: 总损失值
        :return: loss_dict: 各损失函数的损失值字典
        """

        if epoch >= self.max_epoch:
            raise ValueError(f"Epoch {epoch} exceeds max_epoch {self.max_epoch}.")
        # 如果预测值和真实值的通道数不同，则截取最小的通道数
        if y_pred.shape[1] != y_ori.shape[1]:  
            min_channels = min(y_pred.shape[1], y_ori.shape[1])  
            y_pred = y_pred[:, :min_channels, :]  
            y_ori = y_ori[:, :min_channels, :]
        # 计算每个损失
        # losses = {name: loss_fn(y_pred, y_ori) for name, loss_fn in self.loss_functions.items()}
        losses = {}
        for name, loss_fn in self.loss_functions.items():
            if name == "feature_space":
                # 特征空间损失使用特征张量
                losses[name] = loss_fn(kwargs['y_pred_features'], kwargs['y_true_features'])
            else:
                # 检查是否支持 mask 参数
                if 'mask' in loss_fn.forward.__code__.co_varnames:
                    losses[name] = loss_fn(y_pred, y_ori, mask=kwargs.get('mask', None))
                else:
                    losses[name] = loss_fn(y_pred, y_ori)

        total_loss = 0.0
        # 使用不同的字典来追踪不同的值
        raw_losses = {}
        weighted_losses_for_print = {}
        weights_for_print = {}

        for name, loss_val in losses.items():
            raw_losses[name] = loss_val.item()  # 存储原始损失
            if self.learnable:
                log_var = self.log_vars[name]
                clamped_log_var = torch.clamp(log_var, min=-10, max=10)
                weight = torch.exp(-clamped_log_var)
                final_loss_term = weight * loss_val + clamped_log_var
                total_loss += final_loss_term
                weighted_losses_for_print[name] = (weight * loss_val).item()
                weights_for_print[name] = weight.item()
            else:
                current_weight = self.calculate_dynamic_weight(name, epoch)
                total_loss += current_weight * loss_val
                weights_for_print[name] = current_weight
                weighted_losses_for_print[name] = (current_weight * loss_val).item()

        raw_losses['total_loss'] = total_loss.item()

        # print(f"Epoch {epoch} | raw_loss: {raw_losses} | weighted_loss: {weighted_losses_for_print} | weights: {weights_for_print}| log_vars: {log_vars_for_print}")
        # log_vars_for_print = {name: log_var.item() for name, log_var in self.log_vars.items()} if self.learnable else {}

        return total_loss
