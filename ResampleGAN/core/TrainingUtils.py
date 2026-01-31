import logging
import torch
import numpy as np
from typing import List


class TrainingUtils:
    """训练相关的工具函数集合"""

    @staticmethod
    def setup_logger(mode: str = "read_write", log_file: str = "training.log") -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        # 避免重复添加handler
        if logger.handlers:
            logger.handlers.clear()

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # 文件处理器
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # 控制台处理器
        if mode != "write_only":
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        return logger

    @staticmethod
    def check_gradients(parameters: List[torch.nn.Parameter]) -> float:
        """检查梯度范数"""
        total_norm = 0
        valid_params = [p for p in parameters if p.grad is not None]

        for p in valid_params:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2

        return total_norm ** 0.5

    @staticmethod
    def monitor_weights(model: torch.nn.Module) -> float:
        """监控模型权重范数"""
        with torch.no_grad():
            total_weight_norm = sum(p.norm().item() ** 2 for p in model.parameters()) ** 0.5
        return total_weight_norm

    @staticmethod
    def initialize_weights(model: torch.nn.Module):
        """初始化模型权重"""
        for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    @staticmethod
    def set_seed(seed: int = 42):
        """设置随机种子"""
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @staticmethod
    def cleanup_memory():
        """清理GPU内存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()

    @staticmethod
    def count_parameters(model: torch.nn.Module) -> dict:
        """统计模型参数数量"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        return {
            'total': total_params,
            'trainable': trainable_params,
            'non_trainable': total_params - trainable_params
        }


def quick_setup(seed: int = 42, log_file: str = "training.log"):
    """快速设置训练环境"""
    TrainingUtils.set_seed(seed)
    logger = TrainingUtils.setup_logger(log_file=log_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

    return logger, device