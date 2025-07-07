from ResampleGAN.TransGAN.Discriminator import Discriminator
import torch
import os
from typing import Dict, List, Tuple, Optional, Any, Union
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
import logging


class ModelManager:
    """Unified model manager - responsible for all model-related operations"""

    def __init__(self, logger=None, config=None):
        """
        Initialize model manager

        Args:
            config: Training configuration object containing model parameters
        """
        self.config = config
        self.device = getattr(config, 'device', torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.logger = logger if logger else logging.getLogger(__name__)

    # =================== Model Creation ===================

    def create_model(self, model_type: str, **kwargs) -> torch.nn.Module:
        """
        Unified interface for creating models

        Args:
            model_type: Model type ("TCN", "LSTM", "xLSTM", "Transformer", etc.)
            **kwargs: Additional model parameters

        Returns:
            torch.nn.Module: Created model
        """
        if model_type == "TCN":
            return self._create_tcn_model(**kwargs)
        elif model_type in ["LSTM"]:
            return self._create_lstm_model(model_type, **kwargs)
        elif model_type in ["Transformer"]:
            return self._create_transformer_model(**kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def _create_tcn_model(self, **kwargs) -> torch.nn.Module:
        """Create TCN model"""
        from ResampleGAN.models.TCN import TCNModel
        return TCNModel(
            input_dim=kwargs.get('input_dim', self.config.dim_input),
            hidden_dim=kwargs.get('hidden_dim', getattr(self.config, 'hidden_dim', 16)),
            output_dim=kwargs.get('output_dim', self.config.dim_input)
        ).to(self.device)

    def _create_lstm_model(self, model_type: str, **kwargs) -> torch.nn.Module:
        """Create LSTM-type model"""
        if model_type == "xLSTM":
            from ResampleGAN.models.xLSTM import xLSTM
            return xLSTM(
                kwargs.get('input_dim', self.config.dim_input),
                kwargs.get('hidden_dim', getattr(self.config, 'hidden_dim', 16))
            ).to(self.device)
        else:  # LSTM
            from ResampleGAN.models.xLSTM import SimpleLSTM
            return SimpleLSTM(
                input_size=kwargs.get('input_size', self.config.dim_input),
                hidden_size=kwargs.get('hidden_size', getattr(self.config, 'hidden_dim', 16)),
                output_size=kwargs.get('output_size', 1)
            ).to(self.device)

    def _create_transformer_model(self, **kwargs) -> torch.nn.Module:
        """Create Transformer model"""
        from ResampleGAN.TransGAN.Generator import Generator
        return Generator(
            num_layers=kwargs.get('num_layers', self.config.num_layers),
            dim_input=kwargs.get('dim_input', self.config.dim_input),
            dim_attention=kwargs.get('dim_attention', self.config.dim_attention),
            num_heads=kwargs.get('num_heads', self.config.num_heads),
            dim_feedforward=kwargs.get('dim_feedforward', self.config.dim_feedforward),
            dropout=kwargs.get('dropout', self.config.dropout),
            use_noise=kwargs.get('use_noise', getattr(self.config, 'use_noise', False)),
            attention_type=kwargs.get('attention_type', self.config.attention_type),
            restore=kwargs.get('restore', getattr(self.config, 'restore', False)),
            use_mask=kwargs.get('use_mask', getattr(self.config, 'use_mask', False)),
            with_bias=kwargs.get('with_bias', self.config.with_bias)
        ).to(self.device)

    def create_generator(self, model_type: str = "Transformer", **kwargs) -> torch.nn.Module:
        """Create generator - compatibility interface"""
        return self.create_model(model_type, **kwargs)

    def create_discriminator(self, **kwargs) -> torch.nn.Module:
        """Create discriminator"""
        return Discriminator(
            num_layers=kwargs.get('num_layers', self.config.num_layers),
            dim_input=kwargs.get('dim_input', self.config.dim_input),
            dim_attention=kwargs.get('dim_attention', self.config.dim_attention),
            num_heads=kwargs.get('num_heads', self.config.num_heads),
            dim_feedforward=kwargs.get('dim_feedforward', self.config.dim_feedforward),
            dropout=kwargs.get('dropout', self.config.dropout),
            attention_type=kwargs.get('attention_type', self.config.attention_type),
            with_bias=kwargs.get('with_bias', self.config.with_bias)
        ).to(self.device)

    # =================== Loss Function Creation ===================

    def create_loss_functions(self, loss_types: List[str] = None) -> Tuple[torch.nn.Module, ...]:
        """
        Create loss functions

        Args:
            loss_types: List of loss function types, e.g., ["combined", "gan", "mse"]

        Returns:
            Tuple of loss functions
        """
        if loss_types is None:
            loss_types = ["combined", "gan"]

        loss_functions = []

        for loss_type in loss_types:
            if loss_type == "combined":
                from ResampleGAN.TransGAN.LossFunction.CombinedLoss import CombinedLoss
                criterion = CombinedLoss(
                    weights=getattr(self.config, 'weights', {}),
                    max_epoch=getattr(self.config, 'n_epochs', 100),
                    learnable=True
                ).to(self.device)
                loss_functions.append(criterion)

            elif loss_type == "gan":
                criterion = torch.nn.BCEWithLogitsLoss()
                loss_functions.append(criterion)

            elif loss_type == "mse":
                criterion = torch.nn.MSELoss()
                loss_functions.append(criterion)

            elif loss_type == "l1":
                criterion = torch.nn.L1Loss()
                loss_functions.append(criterion)

            else:
                raise ValueError(f"Unsupported loss type: {loss_type}")

        return tuple(loss_functions) if len(loss_functions) > 1 else loss_functions[0]

    # =================== Optimizer Creation ===================

    def create_optimizer(self, parameters: Union[torch.nn.Module, List],
                         optimizer_type: str = None, **kwargs) -> torch.optim.Optimizer:
        """
        Create optimizer

        Args:
            parameters: Model parameters or parameter list
            optimizer_type: Optimizer type
            **kwargs: Optimizer parameters

        Returns:
            torch.optim.Optimizer: Optimizer instance
        """
        # Handle parameters
        if isinstance(parameters, torch.nn.Module):
            params = list(parameters.parameters())
        else:
            params = parameters

        # Get configuration
        optimizer_type = optimizer_type or getattr(self.config, 'optimizer_type', 'AdamW')
        lr = kwargs.get('lr', getattr(self.config, 'lr', 1e-3))
        weight_decay = kwargs.get('weight_decay', getattr(self.config, 'weight_decay', 1e-3))

        if optimizer_type == 'AdamW':
            return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, **kwargs)
        elif optimizer_type == 'Adam':
            return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay, **kwargs)
        elif optimizer_type == 'SGD':
            return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay,
                                   momentum=kwargs.get('momentum', 0.9), **kwargs)
        elif optimizer_type == 'RMSprop':
            return torch.optim.RMSprop(params, lr=lr, weight_decay=weight_decay, **kwargs)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    def create_optimizers_for_gan(self, generator: torch.nn.Module, discriminator: torch.nn.Module,
                                  criterion_combined: torch.nn.Module = None) -> Tuple:
        """
        Create optimizers for GAN

        Args:
            generator: Generator model
            discriminator: Discriminator model
            criterion_combined: Combined loss function (may contain learnable parameters)

        Returns:
            Tuple: (optimizer_G, optimizer_D, scheduler_G, scheduler_D, params_G)
        """
        # Generator parameters
        params_G = list(generator.parameters())
        if criterion_combined and hasattr(criterion_combined, 'learnable') and criterion_combined.learnable:
            params_G += list(criterion_combined.parameters())

        # Create optimizers
        optimizer_G = self.create_optimizer(params_G)
        optimizer_D = self.create_optimizer(discriminator)

        # Create schedulers
        scheduler_G = self.create_scheduler(optimizer_G)
        scheduler_D = self.create_scheduler(optimizer_D)

        return optimizer_G, optimizer_D, scheduler_G, scheduler_D, params_G

    # =================== Learning Rate Scheduler Creation ===================

    def create_scheduler(self, optimizer: torch.optim.Optimizer,
                         scheduler_type: str = None, **kwargs) -> Optional[Union[_LRScheduler, ReduceLROnPlateau]]:
        """
        Create learning rate scheduler

        Args:
            optimizer: Optimizer
            scheduler_type: Scheduler type
            **kwargs: Scheduler parameters

        Returns:
            Learning rate scheduler instance
        """
        scheduler_type = scheduler_type or getattr(self.config, 'scheduler_type', 'WarmupCosine')

        if scheduler_type == 'ReduceLROnPlateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=kwargs.get('mode', getattr(self.config, 'mode', 'min')),
                factor=kwargs.get('factor', getattr(self.config, 'factor', 0.5)),
                patience=kwargs.get('patience', getattr(self.config, 'patience', 5))
            )

        elif scheduler_type == 'WarmupLinear':
            return get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=kwargs.get('warmup_steps', getattr(self.config, 'warmup_steps', 100)),
                num_training_steps=kwargs.get('total_steps', getattr(self.config, 'total_steps', 1000))
            )

        elif scheduler_type == 'WarmupCosine':
            return get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=kwargs.get('warmup_steps', getattr(self.config, 'warmup_steps', 100)),
                num_training_steps=kwargs.get('total_steps', getattr(self.config, 'total_steps', 1000))
            )
        elif scheduler_type is None or scheduler_type.lower() == 'none':
            return None
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    # =================== Model Management ===================

    def load_model(self, model: torch.nn.Module, model_path: str, strict: bool = True) -> torch.nn.Module:
        """
        Load model weights

        Args:
            model: Model instance
            model_path: Model weight file path
            strict: Whether to strictly match parameter names

        Returns:
            Model with loaded weights
        """
        if os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                model.load_state_dict(state_dict, strict=strict)
                self.logger.info(f"Loaded model from {model_path}")
            except Exception as e:
                self.logger.error(f"Failed to load model from {model_path}: {e}")
                raise
        else:
            raise FileNotFoundError(f"Model path {model_path} does not exist")
        return model

    def save_model(self, model: torch.nn.Module, save_path: str,
                   create_dir: bool = True, additional_info: Dict = None):
        """
        Save model weights

        Args:
            model: Model instance
            save_path: Save path
            create_dir: Whether to create directory
            additional_info: Additional information (such as epoch, loss, etc.)
        """
        if create_dir:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

        save_dict = {'model_state_dict': model.state_dict()}
        if additional_info:
            save_dict.update(additional_info)

        torch.save(save_dict, save_path)
        self.logger.info(f"Model saved to {save_path}")

    def load_checkpoint(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                        checkpoint_path: str) -> Tuple[torch.nn.Module, torch.optim.Optimizer, Dict]:
        """
        Load complete checkpoint (including model, optimizer state, etc.)

        Args:
            model: Model instance
            optimizer: Optimizer instance
            checkpoint_path: Checkpoint path

        Returns:
            Tuple: (model, optimizer, checkpoint_info)
        """
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            info = {k: v for k, v in checkpoint.items()
                    if k not in ['model_state_dict', 'optimizer_state_dict']}

            self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
            return model, optimizer, info
        else:
            self.logger.warning(f"Checkpoint {checkpoint_path} not found, returning unmodified model and optimizer")
            return model, optimizer, {}

    # =================== Model Utility Functions ===================

    def count_parameters(self, model: torch.nn.Module) -> Dict[str, int]:
        """Count model parameters"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        return {
            'total': total_params,
            'trainable': trainable_params,
            'non_trainable': total_params - trainable_params
        }

    def initialize_weights(self, model: torch.nn.Module, method: str = 'xavier_uniform'):
        """
        Initialize model weights

        Args:
            model: Model instance
            method: Initialization method
        """

        def init_func(m):
            if isinstance(m, torch.nn.Linear):
                if method == 'xavier_uniform':
                    torch.nn.init.xavier_uniform_(m.weight)
                elif method == 'xavier_normal':
                    torch.nn.init.xavier_normal_(m.weight)
                elif method == 'kaiming_uniform':
                    torch.nn.init.kaiming_uniform_(m.weight)
                elif method == 'kaiming_normal':
                    torch.nn.init.kaiming_normal_(m.weight)
                else:
                    torch.nn.init.normal_(m.weight, 0, 0.02)

                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                torch.nn.init.ones_(m.weight)
                torch.nn.init.zeros_(m.bias)

        model.apply(init_func)
        self.logger.debug(f"Initialized model weights using {method}")

    def freeze_model(self, model: torch.nn.Module, freeze: bool = True):
        """Freeze or unfreeze model parameters"""
        for param in model.parameters():
            param.requires_grad = not freeze
        status = "frozen" if freeze else "unfrozen"
        self.logger.debug(f"Model parameters {status}")

    def get_model_size(self, model: torch.nn.Module) -> float:
        """Get model size (MB)"""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()

        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb

    def print_model_summary(self, model: torch.nn.Module, model_name: str = "Model"):
        """Print model summary information"""
        params_info = self.count_parameters(model)
        size_mb = self.get_model_size(model)

        self.logger.info(f"   {model_name} Summary")
        self.logger.info(f"   Total parameters: {params_info['total']:,}")
        self.logger.info(f"   Trainable parameters: {params_info['trainable']:,}")
        self.logger.info(f"   Non-trainable parameters: {params_info['non_trainable']:,}")
        self.logger.info(f"   Model size: {size_mb:.2f} MB")


# Usage example
if __name__ == "__main__":
    from dataclasses import dataclass
    import logging


    @dataclass
    class DemoConfig:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dim_input = 1
        dim_attention = 128
        num_heads = 4
        num_layers = 6
        dim_feedforward = 128
        dropout = 0.1
        with_bias = False
        attention_type = ["original"] * 3
        lr = 1e-3
        weight_decay = 1e-3
        optimizer_type = 'AdamW'
        scheduler_type = 'WarmupCosine'
        n_epochs = 100
        warmup_steps = 100
        total_steps = 1000
        weights = {'mse': 1.0}


    # Create manager
    config = DemoConfig()
    logger = logging.getLogger(__name__)
    manager = ModelManager(logger, config)

    # Create models
    generator = manager.create_generator()
    discriminator = manager.create_discriminator()

    # Print model information
    manager.print_model_summary(generator, "Generator")
    manager.print_model_summary(discriminator, "Discriminator")

    # Create loss functions
    criterion_combined, criterion_gan = manager.create_loss_functions(["combined", "gan"])

    # Create optimizers
    optimizer_G, optimizer_D, scheduler_G, scheduler_D, params_G = \
        manager.create_optimizers_for_gan(generator, discriminator, criterion_combined)

    print("All components created successfully!")