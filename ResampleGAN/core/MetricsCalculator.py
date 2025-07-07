from typing import Dict, List, Tuple
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class MetricsCalculator:
    """Metrics calculator - responsible for calculating various evaluation metrics"""

    @staticmethod
    def calculate_discriminator_metrics(predictions: List[int], labels: List[int]) -> Dict[str, float]:
        """Calculate discriminator-related metrics"""
        return {
            'acc': accuracy_score(labels, predictions),
            'precision': precision_score(labels, predictions, zero_division=0),
            'recall': recall_score(labels, predictions, zero_division=0),
            'f1': f1_score(labels, predictions, zero_division=0)
        }

    @staticmethod
    def process_discriminator_outputs(real_preds, fake_preds, fake_initials=None) -> Tuple[List[int], List[int]]:
        """Process discriminator outputs, return prediction and label lists"""
        real_preds_bin = (torch.sigmoid(real_preds).cpu() > 0.5).int().view(-1).tolist()
        fake_preds_bin = (torch.sigmoid(fake_preds).cpu() > 0.5).int().view(-1).tolist()

        all_preds = real_preds_bin + fake_preds_bin
        all_labels = [1] * len(real_preds_bin) + [0] * len(fake_preds_bin)

        if fake_initials is not None:
            fake_initials_bin = (torch.sigmoid(fake_initials).cpu() > 0.5).int().view(-1).tolist()
            all_preds.extend(fake_initials_bin)
            all_labels.extend([0] * len(fake_initials_bin))

        return all_preds, all_labels