import torch
import torch.nn as nn
import numpy as np
from torch import nn
import torch.nn.functional as F

from ResampleGAN.TransGAN.LossFunction.LossRegistry import LossRegistry

loss_registry = LossRegistry()

class BaseLoss(nn.Module):
    """
    损失函数基类, 所有自定义损失函数的父类
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def _align_temporal_scale(self, y_pred, y_ori, with_norm=False):
        """
        对齐时间尺度，返回高分辨率和低分辨率序列

        :param y_pred: 预测值 [batch_size, pred_length]
        :param y_ori: 真实值 [batch_size, true_length]
        :return: (high_res, low_res, scale_factor)
        """
        pred_len = y_pred.shape[1]
        true_len = y_ori.shape[1]

        # 计算时间尺度比例
        scale_factor = max(pred_len, true_len) // min(pred_len, true_len)
        (high_res, low_res) = (y_pred, y_ori) if pred_len > true_len else (y_ori, y_pred)

        # 归一化处理
        if with_norm:
            high_res = (high_res - high_res.mean(dim=1, keepdim=True)) / (high_res.std(dim=1, keepdim=True) + 1e-8)
            low_res = (low_res - low_res.mean(dim=1, keepdim=True)) / (low_res.std(dim=1, keepdim=True) + 1e-8)

        return high_res, low_res, scale_factor, pred_len > true_len

    def forward(self, y_pred, y_ori):

        return self.mse(y_pred, y_ori)


@loss_registry.register("mse", default_weight=1, description="Cross Temporal scale MSE loss for different resolution time series")
class TemporalScaleMSELoss(BaseLoss):
    """
    计算时间尺度MSE损失

    :param y_pred: 预测值 [batch_size, pred_length]
    :param y_ori: 真实值 [batch_size, true_length]
    :return: MSE损失值
    """

    def forward(self, y_pred, y_ori):
        high_res, low_res, scale_factor, is_pred_higher = self._align_temporal_scale(y_pred, y_ori, with_norm=True)

        windows = high_res.unfold(dimension=1, size=scale_factor, step=scale_factor)
        sampled = windows.mean(dim=3)

        return torch.mean((sampled - low_res) ** 2)

@loss_registry.register("mae", default_weight=0, description="Cross Temporal scale MSE loss for different resolution time series")
class TemporalScaleMSELoss(BaseLoss):
    """
    计算时间尺度平均的绝对损失

    :param y_pred: 预测值 [batch_size, pred_length]
    :param y_ori: 真实值 [batch_size, true_length]
    :return: MSE损失值
    """

    def forward(self, y_pred, y_ori):
        high_res, low_res, scale_factor, is_pred_higher = self._align_temporal_scale(y_pred, y_ori, with_norm=True)

        windows = high_res.unfold(dimension=1, size=scale_factor, step=scale_factor)
        sampled = windows.mean(dim=3)

        return torch.mean(torch.abs(sampled - low_res))

@loss_registry.register("L1", default_weight=0, description="Cross Temporal scale Smooth L1 loss for different resolution time series")
class TemporalScaleSmoothL1Loss(BaseLoss):
    """
    计算时间尺度平滑L1损失

    :param y_pred: 预测值 [batch_size, pred_length]
    :param y_ori: 真实值 [batch_size, true_length]
    :param beta: 平滑L1损失的阈值，当误差小于 beta 时为二次惩罚，否则为线性惩罚
    :return: 平滑L1损失值
    """

    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta

    def forward(self, y_pred, y_ori):
        # 对齐时间尺度
        high_res, low_res, scale_factor, is_pred_higher = self._align_temporal_scale(y_pred, y_ori, with_norm=True)

        # 创建滑动窗口并采样
        windows = high_res.unfold(dimension=1, size=scale_factor, step=scale_factor)
        sampled = windows.mean(dim=3)

        # 计算平滑L1损失
        diff = sampled - low_res
        abs_diff = torch.abs(diff)

        return torch.sum(abs_diff)



@loss_registry.register("spectral", default_weight=0, description="Cross Temporal scale spectral domain loss")
class TemporalScaleSpectralLoss(BaseLoss):
    """
    计算时间尺度频谱损失，包含高频成分的惩罚项
    """

    def forward(self, y_pred, y_ori):
        # 计算FFT
        pred_fft = torch.fft.rfft(y_pred, dim=1)
        ori_fft = torch.fft.rfft(y_ori, dim=1)

        # 获取两个FFT结果的长度
        pred_freq_len = pred_fft.shape[1]
        ori_freq_len = ori_fft.shape[1]
        min_freq_len = min(pred_freq_len, ori_freq_len)

        # 计算共同频率范围的损失
        pred_magnitude_common = torch.log1p(pred_fft[:, :min_freq_len].abs() + 1e-8)
        ori_magnitude_common = torch.log1p(ori_fft[:, :min_freq_len].abs() + 1e-8)
        pred_angle_common = torch.angle(pred_fft[:, :min_freq_len])
        ori_angle_common = torch.angle(ori_fft[:, :min_freq_len])

        magnitude_loss = torch.mean((pred_magnitude_common - ori_magnitude_common) ** 2)
        phase_loss = torch.mean((pred_angle_common - ori_angle_common) ** 2)

        # 计算额外高频成分的惩罚
        high_freq_penalty = 0.0
        if pred_freq_len > min_freq_len:
            # 预测值包含额外的高频成分
            extra_freq_magnitude = torch.log1p(pred_fft[:, min_freq_len:].abs() + 1e-8)
            high_freq_penalty = torch.mean(extra_freq_magnitude ** 2)
        elif ori_freq_len > min_freq_len:
            # 真实值包含额外的高频成分
            extra_freq_magnitude = torch.log1p(ori_fft[:, min_freq_len:].abs() + 1e-8)
            high_freq_penalty = torch.mean(extra_freq_magnitude ** 2)

        # 组合所有损失
        total_loss = magnitude_loss + 0.1 * phase_loss + 0 * high_freq_penalty

        return total_loss


@loss_registry.register("kurtosis", default_weight=0, description="Cross Temporal scale kurtosis loss")
class TemporalScaleKurtosisLoss(BaseLoss):
    """
    计算时间尺度峰度损失

    :param y_pred: 预测值 [batch_size, pred_length]
    :param y_ori: 真实值 [batch_size, true_length]
    :return: 峰度损失值
    """

    def calculate_kurtosis(self, x):
        """计算峰度"""
        fourth_moment = torch.mean(x ** 4, dim=1)
        return fourth_moment - 3

    def forward(self, y_pred, y_ori):
        high_res, low_res, _, _ = self._align_temporal_scale(y_pred, y_ori, with_norm=True)
        high_kurtosis = self.calculate_kurtosis(high_res)
        low_kurtosis = self.calculate_kurtosis(low_res)
        return torch.mean((high_kurtosis - low_kurtosis) ** 2)


@loss_registry.register("skewness", default_weight=0, description="Cross Temporal scale skewness loss")
class TemporalScaleSkewnessLoss(BaseLoss):
    """
    计算时间尺度偏度损失

    :param y_pred: 预测值 [batch_size, pred_length]
    :param y_ori: 真实值 [batch_size, true_length]
    :return: 偏度损失值
    """

    def calculate_skewness(self, x):
        """计算偏度"""
        return torch.mean(x ** 3, dim=1)

    def forward(self, y_pred, y_ori):
        high_res, low_res, _, _ = self._align_temporal_scale(y_pred, y_ori, with_norm=True)
        high_skewness = self.calculate_skewness(high_res)
        low_skewness = self.calculate_skewness(low_res)
        return torch.mean((high_skewness - low_skewness) ** 2)


@loss_registry.register("autocorr", default_weight=0, description="Cross Temporal scale autocorrelation loss")
class TemporalScaleAutocorrLoss(BaseLoss):
    """
    计算时间尺度自相关损失

    :param y_pred: 预测值 [batch_size, pred_length]
    :param y_ori: 真实值 [batch_size, true_length]
    :return: 自相关损失值
    """

    def __init__(self, max_lag=10):
        super().__init__()
        self.max_lag = max_lag

    def calculate_autocorr(self, x, lag):
        """
        计算给定lag的自相关系数

        :param x: 输入张量 [batch_size, sequence_length]
        :param lag: 滞后阶数
        :return: 自相关系数
        """
        return (x[:, :-lag] * x[:, lag:]).mean(dim=1)

    def forward(self, y_pred, y_ori):
        high_res, low_res, _, _ = self._align_temporal_scale(y_pred, y_ori, with_norm=True)

        total_loss = 0
        for lag in range(1, self.max_lag + 1):
            high_autocorr = self.calculate_autocorr(high_res, lag)
            low_autocorr = self.calculate_autocorr(low_res, lag)
            weight = 1.0 / np.sqrt(lag)
            lag_loss = torch.mean((high_autocorr - low_autocorr) ** 2)
            total_loss += weight * lag_loss

        return total_loss / self.max_lag


@loss_registry.register("smoothness", default_weight=0, description="Second-order smoothness loss")
class SecondOrderSmoothnessLoss(BaseLoss):
    """
    计算二阶平滑度损失

    :param y_pred: 预测值 [batch_size, pred_length]
    :return: 二阶平滑度损失值
    """

    def calculate_second_order(self, x):
        """
        计算二阶差分

        :param x: 输入序列 [batch_size, length]
        :return: second_order differences
        """
        # 计算一阶差分
        first_order = x[:, 1:] - x[:, :-1]  # [batch_size, length-1]
        # 计算二阶差分
        return first_order[:, 1:] - first_order[:, :-1]  # [batch_size, length-2]

    def forward(self, y_pred, y_ori):
        # 只计算预测值的平滑度
        second_order = self.calculate_second_order(y_pred)

        # 计算二阶平滑度损失
        second_order_loss = torch.mean(second_order ** 2)

        return second_order_loss


@loss_registry.register("gradient", default_weight=0, description="First-order difference loss with window alignment")
class FirstOrderDifferenceLoss(BaseLoss):
    """
    计算一阶差分损失，使用窗口对齐比较原始序列和预测序列的差分

    :param y_pred: 预测值 [batch_size, pred_length]
    :param y_ori: 真实值 [batch_size, true_length]
    :return: 一阶差分损失值
    """

    def calculate_first_order(self, x):
        """
        计算一阶差分

        :param x: 输入序列 [batch_size, length]
        :return: first_order differences [batch_size, length-1]
        """
        return x[:, 1:] - x[:, :-1]

    def forward(self, y_pred, y_ori):
        # 计算两个序列的一阶差分
        pred_diff = self.calculate_first_order(y_pred)  # [batch_size, pred_length-1]
        ori_diff = self.calculate_first_order(y_ori)  # [batch_size, true_length-1]

        # 获取两个差分序列的长度
        pred_len = pred_diff.shape[1]
        true_len = ori_diff.shape[1]
        scale_factor = max(pred_len, true_len) // min(pred_len, true_len)
        is_pred_higher = pred_len > true_len

        # 确定高低分辨率序列
        (high_res, low_res) = (pred_diff, ori_diff) if is_pred_higher else (ori_diff, pred_diff)

        # 使用窗口进行下采样
        windows = high_res.unfold(dimension=1, size=scale_factor, step=scale_factor)

        # 计算窗口内的平均差分
        sampled_diff = windows.mean(dim=3)  # 使用mean而不是max/min，因为我们关心的是变化率
        high_diff, low_diff = (sampled_diff, low_res) if is_pred_higher else (low_res, sampled_diff)

        return torch.mean((high_diff - low_diff) ** 2)


@loss_registry.register("mean", default_weight=0, description="Cross Temporal scale mean loss")
class TemporalScaleMeanLoss(BaseLoss):
    """
    计算时间尺度均值损失

    :param y_pred: 预测值 [batch_size, pred_length]
    :param y_ori: 真实值 [batch_size, true_length]
    :return: 均值损失值
    """

    def forward(self, y_pred, y_ori):
        # 使用基类方法对齐时间尺度
        high_res, low_res, scale_factor, is_pred_higher = self._align_temporal_scale(y_pred, y_ori)

        # 使用窗口平均处理高分辨率序列
        windows = high_res.unfold(dimension=1, size=scale_factor, step=scale_factor)
        sampled_high_res = windows.mean(dim=3)

        # 确保使用正确的序列进行比较
        (processed_pred, processed_ori) = (sampled_high_res, low_res) if is_pred_higher else (low_res, sampled_high_res)

        return torch.mean((processed_pred.mean(dim=1) - processed_ori.mean(dim=1)) ** 2)


@loss_registry.register("max", default_weight=0, description="Cross Temporal scale maximum value loss")
class TemporalScaleMaxLoss(BaseLoss):
    """
    计算时间尺度最大值损失

    :param y_pred: 预测值 [batch_size, pred_length]
    :param y_ori: 真实值 [batch_size, true_length]
    :return: 最大值损失值
    """

    def forward(self, y_pred, y_ori):
        high_res, low_res, scale_factor, is_pred_higher = self._align_temporal_scale(y_pred, y_ori)

        # 使用最大值而不是平均值进行下采样
        windows = high_res.unfold(dimension=1, size=scale_factor, step=scale_factor)
        sampled_high_res = windows.max(dim=2)[0]  # 取最大值

        (processed_pred, processed_ori) = (sampled_high_res, low_res) if is_pred_higher else (low_res, sampled_high_res)

        # 计算全局最大值损失
        pred_max = processed_pred.max(dim=1)[0]
        ori_max = processed_ori.max(dim=1)[0]

        return torch.mean((pred_max - ori_max) ** 2)


@loss_registry.register("min", default_weight=0, description="Cross Temporal scale minimum value loss")
class TemporalScaleMinLoss(BaseLoss):
    """
    计算时间尺度最小值损失

    :param y_pred: 预测值 [batch_size, pred_length]
    :param y_ori: 真实值 [batch_size, true_length]
    :return: 最小值损失值
    """

    def forward(self, y_pred, y_ori):
        high_res, low_res, scale_factor, is_pred_higher = self._align_temporal_scale(y_pred, y_ori)

        # 使用最小值而不是平均值进行下采样
        windows = high_res.unfold(dimension=1, size=scale_factor, step=scale_factor)
        sampled_high_res = windows.min(dim=2)[0]  # 取最小值

        (processed_pred, processed_ori) = (sampled_high_res, low_res) if is_pred_higher else (low_res, sampled_high_res)

        # 计算全局最小值损失
        pred_min = processed_pred.min(dim=1)[0]
        ori_min = processed_ori.min(dim=1)[0]

        return torch.mean((pred_min - ori_min) ** 2)


@loss_registry.register("maxima", default_weight=0, description="Cross Temporal scale maxima alignment loss")
class TemporalScaleMaximaAlignedLoss(BaseLoss):
    def find_maxima_with_indices(self, x):
        """
        找出序列中的极大值点及其索引

        :param x: 输入序列 [batch_size, length]
        :return: (maxima_values, maxima_indices) 每个batch的极大值和对应索引
        """
        # 计算差分
        diff = x[:, 1:] - x[:, :-1]  # [batch_size, length-1]

        # 找出符号变化点（极大值点）
        sign_changes = diff[:, 1:] * diff[:, :-1]  # [batch_size, length-2]
        maxima_mask = (sign_changes < 0) & (diff[:, :-1] > 0)  # [batch_size, length-2]

        batch_size = x.size(0)
        maxima_values = []
        maxima_indices = []

        # 对每个batch分别处理
        for i in range(batch_size):
            # 获取该batch的极大值点位置（注意要加1因为我们用的是中间点）
            indices = torch.nonzero(maxima_mask[i]).squeeze() + 1
            # 获取对应的值
            values = x[i][indices]

            maxima_values.append(values)
            maxima_indices.append(indices)

        return maxima_values, maxima_indices

    def forward(self, y_pred, y_ori):
        """
        计算对齐后的极大值损失

        :param y_pred: 预测值 [batch_size, pred_length]
        :param y_ori: 真实值 [batch_size, true_length]
        :return: 损失值
        """
        # 在原始序列中找到极大值点
        ori_maxima_values, ori_maxima_indices = self.find_maxima_with_indices(y_ori)

        pred_len = y_pred.shape[1]
        true_len = y_ori.shape[1]
        scale_factor = max(pred_len, true_len) // min(pred_len, true_len)
        is_pred_higher = pred_len > true_len

        # 创建预测序列的窗口
        windows = y_pred.unfold(dimension=1, size=scale_factor, step=scale_factor) if is_pred_higher else y_pred

        batch_size = y_pred.size(0)
        total_loss = 0
        valid_batches = 0

        # 对每个batch分别处理
        for b in range(batch_size):
            if len(ori_maxima_indices[b]) > 0:
                if is_pred_higher:
                    # 如果预测值分辨率更高，需要将原始索引映射到windows的索引
                    window_indices = ori_maxima_indices[b]
                    # 获取对应window的最大值
                    pred_values = windows[b][window_indices].max(dim=1)[0]  # [num_peaks]
                else:
                    # 如果预测值分辨率更低，需要将原始索引映射到预测序列的索引
                    pred_indices = ori_maxima_indices[b] // scale_factor
                    # 获取预测序列中对应位置的值
                    pred_values = windows[b][pred_indices]  # [num_peaks]

                # 计算损失
                batch_loss = torch.mean((pred_values - ori_maxima_values[b]) ** 2)
                total_loss += batch_loss
                valid_batches += 1

        # 返回平均损失
        return total_loss / valid_batches if valid_batches > 0 else torch.tensor(0.0, device=y_pred.device)


@loss_registry.register("minima", default_weight=0, description="Cross Temporal scale minima alignment loss")
class TemporalScaleMinimaAlignedLoss(BaseLoss):
    def find_minima_with_indices(self, x):
        """
        找出序列中的极小值点及其索引

        :param x: 输入序列 [batch_size, length]
        :return: (minima_values, minima_indices) 每个batch的极小值和对应索引
        """
        # 计算差分
        diff = x[:, 1:] - x[:, :-1]  # [batch_size, length-1]

        # 找出符号变化点（极小值点）
        sign_changes = diff[:, 1:] * diff[:, :-1]  # [batch_size, length-2]
        minima_mask = (sign_changes < 0) & (diff[:, :-1] < 0)  # [batch_size, length-2]  # 注意这里改为 < 0

        batch_size = x.size(0)
        minima_values = []
        minima_indices = []

        # 对每个batch分别处理
        for i in range(batch_size):
            # 获取该batch的极小值点位置（注意要加1因为我们用的是中间点）
            indices = torch.nonzero(minima_mask[i]).squeeze() + 1
            # 获取对应的值
            values = x[i][indices]

            minima_values.append(values)
            minima_indices.append(indices)

        return minima_values, minima_indices

    def forward(self, y_pred, y_ori):
        """
        计算对齐后的极小值损失

        :param y_pred: 预测值 [batch_size, pred_length]
        :param y_ori: 真实值 [batch_size, true_length]
        :return: 损失值
        """
        # 在原始序列中找到极小值点
        ori_minima_values, ori_minima_indices = self.find_minima_with_indices(y_ori)

        pred_len = y_pred.shape[1]
        true_len = y_ori.shape[1]
        scale_factor = max(pred_len, true_len) // min(pred_len, true_len)
        is_pred_higher = pred_len > true_len

        # 创建预测序列的窗口
        windows = y_pred.unfold(dimension=1, size=scale_factor, step=scale_factor) if is_pred_higher else y_pred

        batch_size = y_pred.size(0)
        total_loss = 0
        valid_batches = 0

        # 对每个batch分别处理
        for b in range(batch_size):
            if len(ori_minima_indices[b]) > 0:
                if is_pred_higher:
                    # 如果预测值分辨率更高，需要将原始索引映射到windows的索引
                    window_indices = ori_minima_indices[b]
                    # 获取对应window的最小值
                    pred_values = windows[b][window_indices].min(dim=1)[0]  # [num_valleys]  # 注意这里改为min
                else:
                    # 如果预测值分辨率更低，需要将原始索引映射到预测序列的索引
                    pred_indices = ori_minima_indices[b] // scale_factor
                    # 获取预测序列中对应位置的值
                    pred_values = windows[b][pred_indices]  # [num_valleys]

                # 计算损失
                batch_loss = torch.mean((pred_values - ori_minima_values[b]) ** 2)
                total_loss += batch_loss
                valid_batches += 1

        # 返回平均损失
        return total_loss / valid_batches if valid_batches > 0 else torch.tensor(0.0, device=y_pred.device)


@loss_registry.register("variance", default_weight=0, description="Cross Temporal scale variance loss")
class TemporalScaleVarianceLoss(BaseLoss):
    """
    计算时间尺度方差损失，直接比较全局方差

    :param y_pred: 预测值 [batch_size, pred_length]
    :param y_ori: 真实值 [batch_size, true_length]
    :return: 方差损失值
    """

    def forward(self, y_pred, y_ori):
        # 直接计算序列方差
        pred_var = torch.var(y_pred, dim=1)
        ori_var = torch.var(y_ori, dim=1)

        return torch.mean((pred_var - ori_var) ** 2)


@loss_registry.register("boundary", default_weight=0, description="Cross Temporal scale boundary-aware loss")
class TemporalScaleMSELoss(BaseLoss):
    """
    计算时间尺度MSE损失，对边界区域施加更高的权重

    :param y_pred: 预测值 [batch_size, pred_length]
    :param y_ori: 真实值 [batch_size, true_length]
    :return: MSE损失值
    """

    def forward(self, y_pred, y_ori):
        high_res, low_res, scale_factor, is_pred_higher = self._align_temporal_scale(y_pred, y_ori, with_norm=True)

        # 创建窗口并采样
        windows = high_res.unfold(dimension=1, size=scale_factor, step=scale_factor)
        sampled = windows.mean(dim=3)

        # 计算平方差
        diff = torch.abs(sampled - low_res)

        # 生成权重张量
        weights = torch.zeros_like(diff)  # 默认所有点权重为 0
        weights[:, :int(len(weights)*0.1)] = 1  # 起始 10 个点权重设为 10
        weights[:, -int(len(weights)*0.1):] = 1  # 结束 10 个点权重设为 10

        # 计算加权损失
        weighted_loss = torch.mean(weights * diff)

        return weighted_loss

@loss_registry.register("feature_space", default_weight=0.5, description="Feature space L1 loss between real and generated features")
class FeatureSpaceLoss(BaseLoss):
    def forward(self, fake_features, real_features):

        loss = F.mse_loss(fake_features.mean(dim=0), real_features.mean(dim=0))

        # 可选：也匹配方差（二阶矩）
        loss += F.mse_loss(fake_features.std(dim=0), real_features.std(dim=0))

        return loss
    
@loss_registry.register("cross_corr", default_weight=1, description="Cross-variable correlation loss")
class CrossVariableCorrelationLoss(BaseLoss):
    def forward(self, y_pred, y_ori):
        def corrcoef(x):
            x = x - x.mean(dim=1, keepdim=True)
            cov = torch.bmm(x.transpose(1, 2), x) / (x.size(1) - 1)
            std = x.std(dim=1, keepdim=True) + 1e-8
            return cov / (std.transpose(2, 1) @ std)

        corr_pred = corrcoef(y_pred)
        corr_true = corrcoef(y_ori)
        return F.mse_loss(corr_pred, corr_true)

@loss_registry.register("entropy", default_weight=1, description="Entropy difference loss")
class EntropyDifferenceLoss(BaseLoss):
    def forward(self, y_pred, y_ori):
        def entropy(x):
            std = x.std(dim=1) + 1e-8
            return 0.5 * torch.log(2 * np.pi * np.e * std**2)
        return F.mse_loss(entropy(y_pred), entropy(y_ori))
