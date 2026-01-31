import torch
import numpy as np
from typing import Dict, List
from ResampleGAN.core.Trainer import BaseTrainer


class Phase1Trainer(BaseTrainer):
    """Phase 1训练器 - 生成器预训练（兼容原始train函数）"""

    def init_history(self) -> Dict:
        """Phase 1只需要简单的训练和验证损失"""
        return {
            "train_loss": [],
            "valid_loss": []
        }

    def setup_models(self):
        """设置Phase 1的模型和优化器"""
        # 创建生成器模型
        self.model = self.model_manager.create_generator("Transformer")

        # 初始化权重（如果配置要求）
        if getattr(self.config, 'init_weights', False):
            self.model_manager.initialize_weights(self.model)

        # 创建损失函数
        self.criterion = self.model_manager.create_loss_functions(["combined"])

        # 准备优化器参数
        params = list(self.model.parameters())
        if hasattr(self.criterion, 'learnable') and self.criterion.learnable:
            params += list(self.criterion.parameters())

        # 创建优化器和调度器
        self.optimizer = self.model_manager.create_optimizer(params)
        self.scheduler = self.model_manager.create_scheduler(self.optimizer)

        # 打印模型信息
        self.model_manager.print_model_summary(self.model, "Generator")

        self.logger.info("Phase 1 setup completed")

    def calculate_loss(self, batch, epoch: int = 0):
        """计算Phase 1的损失 - 兼容原始calculate_loss函数"""
        x_input, x_initial, x_initial_mask, x_output, mask, s_in, s_out, batch_size = \
            self.data_processor.process_batch(batch)

        # Phase 1使用Transformer_pretrain模式
        pred = self.model(x_input, x_initial, s_in, s_out, mask, x_initial_mask)
        target = x_input  # Phase 1的目标是重建输入

        loss = self.criterion(pred, target, epoch=epoch)

        return loss, pred.min().item(), pred.max().item()

    def train_epoch(self, train_loader) -> Dict:
        """Phase 1训练一个epoch"""
        self.model.train()
        self.criterion.train()

        total_train_loss = 0
        all_grad_norms = []

        # 获取所有可训练参数
        trainable_params = list(self.model.parameters())
        if hasattr(self.criterion, 'learnable') and self.criterion.learnable:
            trainable_params += list(self.criterion.parameters())

        for batch in train_loader:
            self.optimizer.zero_grad()

            loss, pred_min, pred_max = self.calculate_loss(batch, epoch=0)
            loss.backward()

            # 检查和裁剪梯度
            grad_norm = self.check_gradients(trainable_params)
            all_grad_norms.append(grad_norm)
            torch.nn.utils.clip_grad_norm_(trainable_params, self.config.grad_clip_threshold)

            self.optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_grad_norm = np.mean(all_grad_norms)

        self.logger.info(f"Train gradient norm: {avg_grad_norm:.4f}")

        return {"loss": avg_train_loss}

    def validate_epoch(self, test_loader) -> Dict:
        """Phase 1验证一个epoch"""
        self.model.eval()
        self.criterion.eval()

        valid_loss = 0
        output_min, output_max = float('inf'), float('-inf')

        with torch.no_grad():
            for batch in test_loader:
                val_loss, pred_min, pred_max = self.calculate_loss(batch, epoch=0)
                output_min = min(output_min, pred_min)
                output_max = max(output_max, pred_max)
                valid_loss += val_loss.item()

        avg_valid_loss = valid_loss / len(test_loader)

        self.logger.info(f"Output range: [{output_min:.4f}, {output_max:.4f}]")

        # 显示可学习参数的值
        if hasattr(self.criterion, 'learnable') and self.criterion.learnable:
            if hasattr(self.criterion, 'log_vars'):
                log_var_values = {name: param.item() for name, param in self.criterion.log_vars.items()}
                self.logger.info(f"Loss weights: {log_var_values}")

        return {"loss": avg_valid_loss}

    def check_gradients(self, parameters: List[torch.nn.Parameter]) -> float:
        """检查梯度范数"""
        total_norm = 0
        valid_params = [p for p in parameters if p.grad is not None]

        for p in valid_params:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2

        return total_norm ** 0.5

    def monitor_weights(self, model: torch.nn.Module) -> float:
        """监控模型权重范数"""
        with torch.no_grad():
            total_weight_norm = sum(p.norm().item() ** 2 for p in model.parameters()) ** 0.5
        return total_weight_norm

    def update_schedulers(self):
        """更新学习率调度器"""
        if self.scheduler:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # ReduceLROnPlateau需要传入验证损失，这里需要在validate_epoch后调用
                pass
            else:
                self.scheduler.step()

    def log_epoch_results(self, epoch: int, train_metrics: Dict, valid_metrics: Dict):
        """Phase 1的日志记录"""
        train_loss = train_metrics.get('loss', 0)
        valid_loss = valid_metrics.get('loss', 0)

        self.logger.info(
            f"Epoch {epoch + 1}/{self.config.n_epochs} | "
            f"Train Loss: {train_loss:.6f} | Valid Loss: {valid_loss:.6f}"
        )

        # 记录权重范数
        weight_norm = self.monitor_weights(self.model)
        self.logger.info(f"Model weight norm: {weight_norm:.4f}")

    def save_best_models(self, epoch: int, metrics: Dict):
        """保存最佳模型"""
        current_loss = metrics.get('loss', float('inf'))

        if epoch == 0 or current_loss < min(self.history["valid_loss"][:-1]):
            self.logger.info(f"Save best model (Valid Loss: {current_loss:.4f})")

            filename = self.get_model_filename("generator")
            self.checkpoint_manager.save_checkpoint(self.model, filename)


class Phase2Trainer(BaseTrainer):
    """Phase 2训练器 - 联合训练生成器和判别器"""

    def setup_models(self):
        """设置Phase 2的模型和优化器"""
        # 创建模型
        self.generator = self.model_manager.create_generator()
        self.discriminator = self.model_manager.create_discriminator()

        # 加载预训练权重
        self.load_pretrained_weights()

        # 创建损失函数
        self.criterion_combine, self.criterion_gan = self.model_manager.create_loss_functions(["combined", "gan"])

        # 创建优化器和调度器
        self.optimizer_G, self.optimizer_D, self.scheduler_G, self.scheduler_D, self.params_G = \
            self.model_manager.create_optimizers_for_gan(self.generator, self.discriminator, self.criterion_combine)

        # 打印模型信息
        self.model_manager.print_model_summary(self.generator, "Generator")
        self.model_manager.print_model_summary(self.discriminator, "Discriminator")

        self.logger.info("Phase 2 setup completed")

    def load_pretrained_weights(self):
        """加载Phase 1的生成器权重"""
        generator_path = self.get_previous_phase_path(1, "generator")
        self.generator = self.model_manager.load_model(self.generator, generator_path)

    def train_epoch(self, train_loader) -> Dict:
        """Phase 2的训练逻辑"""
        self.generator.train()
        self.discriminator.train()
        self.criterion_combine.train()

        g_loss_epoch, d_loss_epoch = [], []
        all_preds_epoch, all_labels_epoch = [], []
        total_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            x_input, x_initial, x_initial_mask, x_output, mask, s_in, s_out, batch_size = \
                self.data_processor.process_batch(batch)

            C = x_input.shape[-1]
            real_labels, fake_labels = self.data_processor.create_labels(batch_size, C, self.config.device)

            # 训练判别器
            self.model_manager.freeze_model(self.discriminator, freeze=False)
            self.optimizer_D.zero_grad()

            real_preds, _ = self.discriminator(x_input, s_in, return_features=True)
            d_real_loss = self.criterion_gan(real_preds, real_labels)

            with torch.no_grad():
                fake_output = self.generator(x_input, x_initial, s_in, s_out, mask, x_initial_mask=x_initial_mask)

            fake_preds, _ = self.discriminator(fake_output.detach(), s_out, return_features=True)
            fake_initials, _ = self.discriminator(x_initial.detach(), s_out, return_features=True)
            d_fake_loss = self.criterion_gan(fake_preds, fake_labels)
            d_initial_loss = self.criterion_gan(fake_initials, fake_labels)

            d_loss = (d_real_loss + d_fake_loss + 1 * d_initial_loss)
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.config.grad_clip_threshold)
            self.optimizer_D.step()

            d_loss_epoch.append(d_loss.item())

            # 记录判别器指标
            all_preds, all_labels = self.metrics_calculator.process_discriminator_outputs(
                real_preds, fake_preds, fake_initials)
            all_preds_epoch.extend(all_preds)
            all_labels_epoch.extend(all_labels)

            # 训练生成器
            total_batches += 1
            if total_batches % self.config.critic == 0:
                self.model_manager.freeze_model(self.discriminator, freeze=True)

                self.optimizer_G.zero_grad()
                fake_output = self.generator(x_input, x_initial, s_in, s_out, mask, x_initial_mask=x_initial_mask)
                preds_for_g, fake_features = self.discriminator(fake_output, s_out, return_features=True)
                preds_for_real, real_features = self.discriminator(x_input, s_out, return_features=True)
                g_loss_gan = self.criterion_gan(preds_for_g, real_labels)
                g_loss_recon = self.criterion_combine(fake_output, x_input, 0,
                                                      y_pred_features=fake_features,
                                                      y_true_features=real_features)
                g_loss = 1 * g_loss_recon + 0 * g_loss_gan
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.params_G, self.config.grad_clip_threshold)
                self.optimizer_G.step()
                g_loss_epoch.append(g_loss.item())

        # 计算指标
        avg_g_loss = np.mean(g_loss_epoch) if g_loss_epoch else 0
        avg_d_loss = np.mean(d_loss_epoch)
        train_d_metrics = self.metrics_calculator.calculate_discriminator_metrics(all_preds_epoch, all_labels_epoch)

        return {**train_d_metrics, 'g_loss': avg_g_loss, 'd_loss': avg_d_loss}

    def validate_epoch(self, test_loader) -> Dict:
        """Phase 2验证一个epoch"""
        self.generator.eval()
        self.discriminator.eval()
        self.criterion_combine.eval()

        valid_g_losses = []
        valid_d_losses = []
        valid_all_preds = []
        valid_all_labels = []

        with torch.no_grad():
            for batch in test_loader:
                x_input, x_initial, x_initial_mask, x_output, mask, s_in, s_out, batch_size = \
                    self.data_processor.process_batch(batch)

                C = x_input.shape[-1]
                real_labels, fake_labels = self.data_processor.create_labels(batch_size, C, self.config.device)

                # 判别器评估
                real_preds, _ = self.discriminator(x_input, s_in, return_features=True)
                d_real_loss = self.criterion_gan(real_preds, real_labels)

                fake_output = self.generator(x_input, x_initial, s_in, s_out, mask, x_initial_mask=x_initial_mask)
                fake_preds, _ = self.discriminator(fake_output, s_out, return_features=True)
                fake_initials, _ = self.discriminator(x_initial, s_out, return_features=True)
                d_fake_loss = self.criterion_gan(fake_preds, fake_labels)
                d_initial_loss = self.criterion_gan(fake_initials, fake_labels)

                valid_d_loss = (d_real_loss + d_fake_loss + 0 * d_initial_loss)
                valid_d_losses.append(valid_d_loss.item())

                # 生成器评估
                preds_for_g, fake_features = self.discriminator(fake_output, s_out, return_features=True)
                preds_for_real, real_features = self.discriminator(x_input, s_out, return_features=True)
                g_loss_recon = self.criterion_combine(fake_output, x_input, 0,
                                                      y_pred_features=fake_features,
                                                      y_true_features=real_features)
                valid_g_loss = 1 * g_loss_recon
                valid_g_losses.append(valid_g_loss.item())

                # 计算判别器指标
                all_preds, all_labels = self.metrics_calculator.process_discriminator_outputs(
                    real_preds, fake_preds, fake_initials)
                valid_all_preds.extend(all_preds)
                valid_all_labels.extend(all_labels)

        # 计算平均指标
        avg_valid_g_loss = np.mean(valid_g_losses)
        avg_valid_d_loss = np.mean(valid_d_losses)
        valid_metrics = self.metrics_calculator.calculate_discriminator_metrics(valid_all_preds, valid_all_labels)

        return {**valid_metrics, 'g_loss': avg_valid_g_loss, 'd_loss': avg_valid_d_loss}

    def save_best_models(self, epoch: int, metrics: Dict):
        """保存最佳模型"""
        valid_g_loss = metrics.get('g_loss', float('inf'))
        valid_d_loss = metrics.get('d_loss', float('inf'))

        if epoch == 0 or valid_g_loss < min(self.history["valid_g_loss"][:-1]):
            self.logger.info(f"Save best generator (Valid G_Loss: {valid_g_loss:.4f})")
            filename = self.get_model_filename("generator")
            self.checkpoint_manager.save_checkpoint(self.generator, filename)

        if epoch == 0 or valid_d_loss < min(self.history["valid_d_loss"][:-1]):
            self.logger.info(f"Save best discriminator (Valid D_Loss: {valid_d_loss:.4f})")
            filename = self.get_model_filename("discriminator")
            self.checkpoint_manager.save_checkpoint(self.discriminator, filename)


class Phase3Trainer(BaseTrainer):
    """Phase 3训练器 - 仅训练生成器"""

    def setup_models(self):
        """设置Phase 3的模型和优化器"""
        # 创建模型
        self.generator = self.model_manager.create_generator()
        self.discriminator = self.model_manager.create_discriminator()

        # 加载预训练权重
        self.load_pretrained_weights()

        # 创建损失函数
        self.criterion_combine = self.model_manager.create_loss_functions(["combined"])

        # 创建优化器（只为生成器）
        params_G = list(self.generator.parameters())
        if hasattr(self.criterion_combine, 'learnable') and self.criterion_combine.learnable:
            params_G += list(self.criterion_combine.parameters())

        self.optimizer_G = self.model_manager.create_optimizer(params_G)
        self.scheduler_G = self.model_manager.create_scheduler(self.optimizer_G)
        self.params_G = params_G

        # 打印模型信息
        self.model_manager.print_model_summary(self.generator, "Generator")
        self.model_manager.print_model_summary(self.discriminator, "Discriminator (Frozen)")

        self.logger.info("Phase 3 setup completed")

    def load_pretrained_weights(self):
        """加载Phase 2的权重"""
        generator_path = self.get_previous_phase_path(2, "generator")
        discriminator_path = self.get_previous_phase_path(2, "discriminator")

        self.generator = self.model_manager.load_model(self.generator, generator_path)
        self.discriminator = self.model_manager.load_model(self.discriminator, discriminator_path)

    def train_epoch(self, train_loader) -> Dict:
        """Phase 3的训练逻辑"""
        self.generator.train()
        self.criterion_combine.train()
        self.discriminator.eval()

        # 冻结判别器
        self.model_manager.freeze_model(self.discriminator, freeze=True)

        g_loss_epoch = []

        for batch_idx, batch in enumerate(train_loader):
            x_input, x_initial, x_initial_mask, x_output, mask, s_in, s_out, batch_size = \
                self.data_processor.process_batch(batch)

            # 只训练生成器
            self.optimizer_G.zero_grad()

            fake_output = self.generator(x_input, x_initial, s_in, s_out, mask, x_initial_mask=x_initial_mask)
            preds_for_g, fake_features = self.discriminator(fake_output, s_out, return_features=True)
            preds_for_real, real_features = self.discriminator(x_input, s_out, return_features=True)
            g_loss_recon = self.criterion_combine(fake_output, x_input, 0,
                                                  y_pred_features=fake_features,
                                                  y_true_features=real_features)

            g_loss = 1 * g_loss_recon
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.params_G, self.config.grad_clip_threshold)
            self.optimizer_G.step()

            g_loss_epoch.append(g_loss.item())

        avg_g_loss = np.mean(g_loss_epoch) if g_loss_epoch else 0
        return {'g_loss': avg_g_loss, 'd_loss': 0, 'acc': 0, 'precision': 0, 'recall': 0, 'f1': 0}

    def validate_epoch(self, test_loader) -> Dict:
        """Phase 3验证一个epoch"""
        self.generator.eval()
        self.criterion_combine.eval()
        valid_g_losses = []

        with torch.no_grad():
            for batch in test_loader:
                x_input, x_initial, x_initial_mask, x_output, mask, s_in, s_out, batch_size = \
                    self.data_processor.process_batch(batch)

                fake_output = self.generator(x_input, x_initial, s_in, s_out, mask, x_initial_mask=x_initial_mask)
                preds_for_g, fake_features = self.discriminator(fake_output, s_out, return_features=True)
                preds_for_real, real_features = self.discriminator(x_input, s_out, return_features=True)
                g_loss_recon = self.criterion_combine(fake_output, x_input, 0,
                                                      y_pred_features=fake_features,
                                                      y_true_features=real_features)
                valid_g_loss = 1 * g_loss_recon
                valid_g_losses.append(valid_g_loss.item())

        avg_valid_g_loss = np.mean(valid_g_losses)
        return {'g_loss': avg_valid_g_loss, 'd_loss': 0, 'acc': 0, 'precision': 0, 'recall': 0, 'f1': 0}

    def save_best_models(self, epoch: int, metrics: Dict):
        """保存最佳模型"""
        valid_g_loss = metrics.get('g_loss', float('inf'))

        if epoch == 0 or valid_g_loss < min(self.history["valid_g_loss"][:-1]):
            self.logger.info(f"Save best generator (Valid G_Loss: {valid_g_loss:.4f})")
            filename = self.get_model_filename("generator")
            self.checkpoint_manager.save_checkpoint(self.generator, filename)