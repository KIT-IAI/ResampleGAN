import os
import time
import pickle
from abc import ABC, abstractmethod
from typing import Dict, List
import torch
from ResampleGAN.core.TrainingConfig import TrainingConfig
from ResampleGAN.core.ModelManager import ModelManager
from ResampleGAN.core.DataProcessor import DataProcessor
from ResampleGAN.core.MetricsCalculator import MetricsCalculator


class CheckpointManager:
    """检查点管理器 - 负责模型保存和训练历史记录"""

    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def save_checkpoint(self, model: torch.nn.Module, filename: str):
        """保存模型检查点"""
        save_path = os.path.join(self.save_dir, filename)
        torch.save(model.state_dict(), save_path)

    def save_history(self, history: Dict, filename: str):
        """保存训练历史"""
        save_path = os.path.join(self.save_dir, filename)
        with open(save_path, "wb") as f:
            pickle.dump(history, f)


class BaseTrainer(ABC):
    """基础训练器 - 定义训练的通用流程"""

    def __init__(self, config: TrainingConfig, save_dir: str, logger,
                 key: str, attention: List[int], waveform: str, now: str, base_save_dir: str = None):
        self.config = config
        self.logger = logger
        self.key = key
        self.attention = attention
        self.waveform = waveform
        self.now = now
        self.base_save_dir = base_save_dir or save_dir  # 如果没有提供base_save_dir，使用save_dir

        # 初始化组件
        self.model_manager = ModelManager(logger, config)
        self.data_processor = DataProcessor(config.device)
        self.metrics_calculator = MetricsCalculator()
        self.checkpoint_manager = CheckpointManager(save_dir)

        # 早停相关
        self.best_loss = float('inf')
        self.no_improve_count = 0

        # 初始化历史记录 - 根据不同phase调整
        self.history = self.init_history()

    def init_history(self) -> Dict:
        """初始化历史记录 - 子类可以重写"""
        return {
            "train_loss": [], "valid_loss": [],
            "g_loss": [], "d_loss": [], "d_acc": [], "d_precision": [], "d_recall": [], "d_f1": [],
            "valid_g_loss": [], "valid_d_loss": [], "valid_d_acc": [], "valid_d_precision": [],
            "valid_d_recall": [], "valid_d_f1": []
        }

    @abstractmethod
    def setup_models(self):
        """设置模型和优化器 - 由子类实现"""
        pass

    @abstractmethod
    def train_epoch(self, train_loader) -> Dict:
        """训练一个epoch - 由子类实现"""
        pass

    @abstractmethod
    def validate_epoch(self, test_loader) -> Dict:
        """验证一个epoch - 由子类实现"""
        pass

    def update_schedulers(self):
        """更新学习率调度器 - 默认实现"""
        if hasattr(self, 'scheduler') and self.scheduler:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # 需要传入验证损失
                pass  # 子类需要在调用时传入loss
            else:
                self.scheduler.step()

        if hasattr(self, 'scheduler_G') and self.scheduler_G:
            self.scheduler_G.step()
        if hasattr(self, 'scheduler_D') and self.scheduler_D:
            self.scheduler_D.step()

    def check_early_stopping(self, current_loss: float, epoch: int) -> bool:
        """检查是否需要早停"""
        if not self.config.use_early_stopping:
            return False

        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.no_improve_count = 0
            return False
        else:
            self.no_improve_count += 1
            if self.no_improve_count >= self.config.patience:
                self.logger.info(
                    f"Early stopping at epoch {epoch + 1} (no improvement for {self.config.patience} epochs)")
                return True
        return False

    def log_epoch_results(self, epoch: int, train_metrics: Dict, valid_metrics: Dict):
        """记录epoch结果 - 默认实现，子类可以重写"""
        self.logger.info(f"Epoch {epoch + 1}/{self.config.n_epochs}")

        # 记录训练指标
        train_msg = "Train - "
        for key, value in train_metrics.items():
            if value != 0:  # 只显示非零值
                train_msg += f"{key.upper()}: {value:.4f} | "

        # 记录验证指标
        valid_msg = "Valid - "
        for key, value in valid_metrics.items():
            if value != 0:  # 只显示非零值
                valid_msg += f"{key.upper()}: {value:.4f} | "

        self.logger.info(train_msg.rstrip(" | "))
        self.logger.info(valid_msg.rstrip(" | "))

    @abstractmethod
    def save_best_models(self, epoch: int, metrics: Dict):
        """保存最佳模型 - 由子类实现"""
        pass

    def train(self, train_loader, test_loader):
        """主训练循环 - 通用框架"""
        start_time = time.time()

        # 设置模型
        self.setup_models()

        for epoch in range(self.config.n_epochs):
            # 训练和验证
            train_metrics = self.train_epoch(train_loader)
            valid_metrics = self.validate_epoch(test_loader)

            # 更新历史记录
            self.update_history(train_metrics, valid_metrics)

            # 更新学习率
            self.update_schedulers()

            # 记录结果
            self.log_epoch_results(epoch, train_metrics, valid_metrics)

            # 保存最佳模型
            self.save_best_models(epoch, valid_metrics)

            # 检查早停
            main_loss = valid_metrics.get('loss', valid_metrics.get('g_loss', float('inf')))
            if self.check_early_stopping(main_loss, epoch):
                break

        # 保存训练历史
        self.save_training_history()

        end_time = time.time()
        execution_time = end_time - start_time
        self.logger.info(f"Training completed in {execution_time:.4f} seconds")

        return self.history

    def update_history(self, train_metrics: Dict, valid_metrics: Dict):
        """更新历史记录 - 默认实现"""
        for key, value in train_metrics.items():
            if key in self.history:
                self.history[key].append(value)

        for key, value in valid_metrics.items():
            valid_key = f"valid_{key}" if not key.startswith('valid_') else key
            if valid_key in self.history:
                self.history[valid_key].append(value)

    def save_training_history(self):
        """保存训练历史"""
        if hasattr(self, 'attention') and len(self.attention) >= 3:
            history_filename = f"history_{self.key}_{self.waveform}_self_{self.attention[0]}_conv_{self.attention[1]}_freq_{self.attention[2]}.pkl"
        else:
            history_filename = f"history_{self.key}_{self.waveform}.pkl"

        self.checkpoint_manager.save_history(self.history, history_filename)

    def get_model_filename(self, model_type: str = "generator") -> str:
        """生成模型文件名"""
        if hasattr(self, 'attention') and len(self.attention) >= 3:
            return f"best_{model_type}_{self.key}_{self.waveform}_self_{self.attention[0]}_conv_{self.attention[1]}_freq_{self.attention[2]}.pth"
        else:
            return f"best_{model_type}_{self.key}_{self.waveform}.pth"

    def get_previous_phase_path(self, phase: int, model_type: str = "generator") -> str:
        """获取前一阶段的模型路径"""
        phase_names = {1: "generator", 2: "discriminator"}
        previous_phase_dir = os.path.join(self.base_save_dir, f"{phase}_{phase_names[phase]}")
        filename = self.get_model_filename(model_type)
        return os.path.join(previous_phase_dir, filename)