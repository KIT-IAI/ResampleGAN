import torch
from typing import Tuple

class DataProcessor:
    """Data processor - responsible for batch data processing"""

    def __init__(self, device: torch.device):
        self.device = device

    def process_batch(self, batch) -> Tuple[torch.Tensor, ...]:
        """Process a single batch of data"""
        x_input, x_initial, x_initial_mask, x_output, mask, condition = batch
        x_input = x_input.to(self.device)
        x_initial = x_initial.to(self.device)
        x_initial_mask = x_initial_mask.to(self.device)

        if x_output is not None:
            x_output = x_output.to(self.device)
            batch_size = x_output.size(0)
        else:
            batch_size = x_input.size(0)
            x_output = None

        mask = mask.to(self.device)
        s_in, s_out = condition[0], condition[1]

        return x_input, x_initial, x_initial_mask, x_output, mask, s_in, s_out, batch_size

    def create_labels(self, batch_size: int, num_channels: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create real and fake labels"""
        real_labels = torch.ones(batch_size, num_channels, device=device)
        fake_labels = torch.zeros(batch_size, num_channels, device=device)
        return real_labels, fake_labels