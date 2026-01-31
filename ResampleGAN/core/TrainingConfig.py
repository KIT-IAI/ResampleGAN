import torch
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import os

@dataclass
class TrainingConfig:
    """统一的训练配置类，包含默认值"""

    # 基础配置
    # 注意：device 不适合设置固定默认值，因为需要根据环境判断。
    # 通常在创建实例时传入或通过一个辅助函数获取。
    device: torch.device = field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    n_epochs: int = 120
    batch_size: int = 16
    lr: float = 2e-4
    weight_decay: float = 0.01
    grad_clip_threshold: float = 10.0

    # 模型配置 (整合了ModelConfig_phase_1中的相关配置)
    dim_input: int = 1
    dim_attention: int = 128
    num_heads: int = 4
    dim_feedforward: int = 128
    dropout: float = 0.1
    num_layers: int = 6
    attention_type: List[str] = field(default_factory=list) # 列表中通常不会有默认项，这里给个空列表
    with_bias: bool = False
    use_noise: bool = False # 来自 ModelConfig_phase_1
    restore: bool = False   # 来自 ModelConfig_phase_1
    use_mask: bool = False  # 来自 ModelConfig_phase_1
    init_weights: bool = False # 来自 ModelConfig_phase_1
    hidden_dim: int = 16 # 来自 ModelConfig_phase_1

    # 损失函数权重
    weights: Dict[str, float] = field(default_factory=lambda: {
        'cs_mse': 1.0,
        'cs_smoothness': 1.0,
        'cs_gradient': 1.0,
    })

    # 优化器和调度器
    optimizer_type: str = 'AdamW'
    scheduler_type: str = 'WarmupCosine'
    total_steps: Optional[int] = None # 使用Optional，因为__post_init__会设置
    warmup_steps: Optional[int] = None # 使用Optional，因为__post_init__会设置

    # 训练特定参数
    lambda_gan: float = 0.1
    critic: int = 1
    best_loss: float = float('inf') # 来自 ModelConfig_phase_1
    no_improve_count: int = 0      # 来自 ModelConfig_phase_1
    patience: int = 10             # 来自 ModelConfig_phase_1 (这里使用TrainingConfig中的10)
    use_early_stopping: bool = False # 来自 ModelConfig_phase_1

    # 调度器额外参数 (来自ModelConfig_phase_1，如果scheduler_type是ReduceLROnPlateau则可能需要)
    scheduler_mode: str = 'min'    # 原ModelConfig_phase_1中的'mode'
    scheduler_factor: float = 0.5  # 原ModelConfig_phase_1中的'factor'
    scheduler_patience: int = 5    # 原ModelConfig_phase_1中的'patience' (这里改名以避免与训练patience混淆)


    def __post_init__(self):
        # 确保 device 已经正确初始化
        if isinstance(self.device, str):
            self.device = torch.device(self.device)

        # 计算 total_steps 和 warmup_steps
        if self.total_steps is None:
            self.total_steps = self.n_epochs * 10
        if self.warmup_steps is None:
            self.warmup_steps = int(self.total_steps * 0.1)

        # 检查 attention_type 是否有效，或者根据需要设置默认值
        if not self.attention_type:
            self.attention_type = ["self"]


@dataclass
class ModelPaths:
    """模型路径管理器"""
    base_dir: str
    experiment_id: str
    key: str
    waveform: str
    attention: List[int]

    def __post_init__(self):
        self.attention_str = f"self_{self.attention[0]}_conv_{self.attention[1]}_freq_{self.attention[2]}"
        self.model_filename = f"best_generator_{self.key}_{self.waveform}_{self.attention_str}.pth"
        self.discriminator_filename = f"best_discriminator_{self.key}_{self.waveform}_{self.attention_str}.pth"
        self.history_filename = f"history_{self.key}_{self.waveform}_{self.attention_str}.pkl"
        self.losses_filename = f"losses_{self.key}_{self.waveform}_{self.attention_str}.pkl"

    def get_phase_dir(self, phase: int) -> str:
        """获取指定阶段的目录"""
        phase_names = {1: "generator", 2: "discriminator", 3: "generator"}
        return os.path.join(self.base_dir, f"{phase}_{phase_names[phase]}")

    def get_generator_path(self, phase: int) -> str:
        """获取生成器模型路径"""
        return os.path.join(self.get_phase_dir(phase), self.model_filename)

    def get_discriminator_path(self, phase: int) -> str:
        """获取判别器模型路径"""
        return os.path.join(self.get_phase_dir(phase), self.discriminator_filename)

    def get_history_path(self, phase: int) -> str:
        """获取训练历史路径"""
        return os.path.join(self.get_phase_dir(phase), self.history_filename)

    def get_losses_path(self, phase: int) -> str:
        """获取损失历史路径"""
        return os.path.join(self.get_phase_dir(phase), self.losses_filename)

    def ensure_phase_dir(self, phase: int):
        """确保阶段目录存在"""
        phase_dir = self.get_phase_dir(phase)
        os.makedirs(phase_dir, exist_ok=True)
        return phase_dir