import numpy as np
from scipy.fftpack import fft
from scipy.stats import pearsonr
from scipy.stats import wasserstein_distance

def compute_fft_error(y_true, y_pred):
    """Calculate FFT spectrum error"""
    fft_true = np.abs(fft(y_true, axis=-1))
    fft_pred = np.abs(fft(y_pred, axis=-1))
    return np.mean(np.abs(fft_true - fft_pred))  # Calculate spectrum error

def compute_metrics(y_true, y_pred):
    """Calculate MSE, MAE, RMSE, Pearson correlation, spectrum error"""

    if y_true.shape[-1] != 1:
        rmses, mags, phases, pccs = [], [], [], []
        for i in range(y_true.shape[-1]):
            mse = np.mean((y_true[:,i] - y_pred[:,i]) ** 2)
            rmses.append(np.sqrt(mse))
            mag, phase = compute_advanced_spectral_metrics(y_true[:,i], y_pred[:,i])
            mags.append(mag)
            phases.append(phase)
            # wasserstein_dist = wasserstein_distance(y_true.flatten(), y_pred.flatten())
            pcc, _ = pearsonr(y_true[:,i].flatten(), y_pred[:,i].flatten())
            pccs.append(pcc)
        rmse = np.mean(rmses)
        mag = np.mean(mags)
        phase = np.mean(phases)
        pcc = np.mean(pccs)
    else:
        mse = np.mean((y_true - y_pred) ** 2)
        # mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(mse)
        # fft_error = compute_fft_error(y_true, y_pred)
        mag, phase = compute_advanced_spectral_metrics(y_true, y_pred)
        # wasserstein_dist = wasserstein_distance(y_true.flatten(), y_pred.flatten())
        pcc, _ = pearsonr(y_true.flatten(), y_pred.flatten())
    return rmse, pcc, mag, phase

def compute_advanced_spectral_metrics(y_true, y_pred):
    """
    Calculate an advanced frequency domain evaluation metric aligned with TemporalScaleSpectralLoss.

    Returns a dictionary containing multiple values for detailed analysis.
    """
    # 1. Use rfft to maintain consistency with PyTorch version and be more efficient for real inputs
    # np.fft.rfft computes the Fourier transform for real inputs
    pred_fft = np.fft.rfft(y_pred, axis=-1)
    true_fft = np.fft.rfft(y_true, axis=-1)

    # 2. Handle inputs of different lengths
    pred_freq_len = pred_fft.shape[-1]
    true_freq_len = true_fft.shape[-1]
    min_freq_len = min(pred_freq_len, true_freq_len)

    # Truncate to common frequency length
    pred_fft_common = pred_fft[..., :min_freq_len]
    true_fft_common = true_fft[..., :min_freq_len]

    # 3. Calculate mean squared error of log-magnitude (MSE of Log-Magnitude)
    pred_magnitude = np.log1p(np.abs(pred_fft_common))  # Use log1p for logarithmic scaling
    true_magnitude = np.log1p(np.abs(true_fft_common))
    magnitude_error = np.mean((pred_magnitude - true_magnitude) ** 2)

    # 4. Calculate mean squared error of phase (MSE of Phase)
    pred_angle = np.angle(pred_fft_common)
    true_angle = np.angle(true_fft_common)
    # Direct calculation of difference is usually sufficient for metrics
    phase_error = np.mean((pred_angle - true_angle) ** 2)

    return magnitude_error, phase_error