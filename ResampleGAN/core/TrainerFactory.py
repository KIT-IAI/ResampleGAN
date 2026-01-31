from typing import List
from ResampleGAN.core.Trainer import BaseTrainer
from ResampleGAN.core.PhaseTrainer import Phase1Trainer, Phase2Trainer, Phase3Trainer
from ResampleGAN.core.TrainingConfig import TrainingConfig

class TrainerFactory:
    """训练器工厂类 - 支持所有阶段"""

    @staticmethod
    def create_trainer(phase: int, config: TrainingConfig, save_dir: str, logger,
                       key: str, attention: List[int], waveform: str, now: str,
                       base_save_dir: str = None) -> BaseTrainer:
        """根据阶段创建对应的训练器"""
        if phase == 1:
            return Phase1Trainer(config, save_dir, logger, key, attention, waveform, now, base_save_dir)
        elif phase == 2:
            return Phase2Trainer(config, save_dir, logger, key, attention, waveform, now, base_save_dir)
        elif phase == 3:
            return Phase3Trainer(config, save_dir, logger, key, attention, waveform, now, base_save_dir)
        else:
            raise ValueError(f"Unsupported phase: {phase}")