import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from scipy.interpolate import interp1d
import random
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel, Matern


def normalize_pair(x_input, x_output, minmax_scaler):
    """
    Normalize input and output pairs using the same scaler to ensure consistency

    Args:
        x_input (numpy.ndarray): Input time series data
        x_output (numpy.ndarray): Output time series data
        minmax_scaler (tuple): Min-max scaling range (e.g., (-1, 1))

    Returns:
        tuple: Normalized (x_input, x_output) arrays
    """
    # Create MinMaxScaler with specified range
    scaler = MinMaxScaler(feature_range=minmax_scaler)

    # Fit scaler on combined input and output data to ensure same scaling
    scaler.fit(x_input)

    # Apply same transformation to both input and output
    return scaler.transform(x_input), scaler.transform(x_output), scaler


def interpolate_sequence(x_input, input_length, output_length, kind='linear'):
    """
    Interpolate a sequence from input_length to output_length using specified method

    Args:
        x_input (numpy.ndarray): Input sequence data with shape (input_length, features)
        input_length (int): Original sequence length
        output_length (int): Target sequence length after interpolation
        kind (str): Interpolation method ('linear', 'cubic', 'quadratic', 'slinear')

    Returns:
        numpy.ndarray: Interpolated sequence with shape (output_length, features)
    """
    # Original time points
    x = np.arange(input_length)
    # New time points for interpolation
    x_new = np.linspace(0, input_length - 1, output_length)

    # Initialize output array
    x_initial = np.zeros((output_length, x_input.shape[1]))

    # Interpolate each feature column separately
    for c in range(x_input.shape[1]):
        f = interp1d(x, x_input[:, c], kind=kind)
        x_initial[:, c] = f(x_new)

    return x_initial


def interpolate_sequence_gp(x_input, input_length, output_length, kernel=None, alpha=0, normalize_y=True):
    """
    Interpolate a sequence from input_length to output_length using Gaussian Process regression

    Args:
        x_input (numpy.ndarray): Input sequence data with shape (input_length, features)
        input_length (int): Original sequence length
        output_length (int): Target sequence length after interpolation
        kernel: Gaussian Process kernel (default: RBF with optimized hyperparameters)
        alpha (float): Noise level parameter for GP (regularization)
        normalize_y (bool): Whether to normalize target values before fitting

    Returns:
        numpy.ndarray: Interpolated sequence with shape (output_length, features)
    """
    # Original time points (normalized to [0, 1] for better GP performance)
    x = np.linspace(0, 1, input_length).reshape(-1, 1)
    # New time points for interpolation
    x_new = np.linspace(0, 1, output_length).reshape(-1, 1)

    # Initialize output array
    x_initial = np.zeros((output_length, x_input.shape[1]))

    # Default kernel if not provided: RBF with automatic hyperparameter optimization
    if kernel is None:
        # Composite kernel: constant * RBF + white noise
        # This provides flexibility in modeling both smooth patterns and noise
        kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-3, 1e3), nu=0.5)

    # Interpolate each feature column separately using GP
    for c in range(x_input.shape[1]):
        # Get the values for this feature
        y = x_input[:, c].reshape(-1, 1)

        # Create and fit the Gaussian Process model
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            normalize_y=normalize_y,
            n_restarts_optimizer=2  # Multiple restarts for better hyperparameter optimization
        )

        # Fit the GP model to the original points
        gp.fit(x, y)

        # Predict at new points (use only mean prediction)
        y_pred, _ = gp.predict(x_new, return_std=True)

        # Store the interpolated values
        x_initial[:, c] = y_pred.flatten()

    return x_initial

def downsample_sequence(x_input, input_length, output_length):
    """
    Downsample a sequence by averaging values within windows

    Args:
        x_input (numpy.ndarray): Input sequence data
        input_length (int): Original sequence length
        output_length (int): Target sequence length after downsampling

    Returns:
        numpy.ndarray: Downsampled sequence
    """
    # Calculate window size for averaging
    window_size = int(input_length / output_length)
    x_initial = []

    # Process each window
    for i in range(output_length):
        start_idx = i * window_size
        end_idx = min((i + 1) * window_size, input_length)

        # Extract window data and compute mean
        window_data = x_input[start_idx:end_idx]
        window_mean = np.mean(window_data, axis=0)
        x_initial.append(window_mean)

    return np.array(x_initial)


def build_causal_mask(output_length, input_length):
    """
    Build a causal attention mask for transformer-like models
    Ensures that each output position can only attend to input positions
    up to a certain point based on temporal causality

    Args:
        output_length (int): Length of output sequence
        input_length (int): Length of input sequence

    Returns:
        torch.FloatTensor: Causal mask with shape (1, 1, output_length, input_length)
    """
    # Initialize mask with negative infinity (masked positions)
    mask = np.full((output_length, input_length), float('-inf'))

    # Calculate ratio between input and output lengths
    ratio = input_length / output_length

    # For each output position, determine which input positions are visible
    for i in range(output_length):
        # Calculate cutoff index based on causal constraint
        cutoff_idx = int(np.ceil((i + 1) * ratio))
        # Set visible positions to 0 (unmasked)
        mask[i, :cutoff_idx] = 0.0

    # Convert to PyTorch tensor and add batch/head dimensions
    return torch.FloatTensor(mask).unsqueeze(0).unsqueeze(0)


def get_aligned_input_output(df, s_in, s_out):
    """
    Generate aligned input and output dataframes from high-frequency original data

    Args:
        df (pd.DataFrame): Original high-frequency time series data
        s_in (str): Input sampling frequency (e.g., "15min")
        s_out (str): Output sampling frequency (e.g., "5min")

    Returns:
        tuple: (df_input, df_output) - Aligned input and output dataframes

    Note:
        - df_output: Resampled according to s_out frequency
        - df_input: Downsampled from df_output to ensure x_input ≈ x_output[::ratio]
    """
    # Resample to output frequency and forward fill missing values
    df_output = df.resample(s_out).first().ffill()

    # Calculate downsampling ratio
    ratio = int(pd.Timedelta(s_in) / pd.Timedelta(s_out))

    # Ensure consistent datetime index
    df_output.index = pd.date_range(df_output.index[0], periods=len(df_output), freq=s_out)

    # Create input dataframe by downsampling output dataframe
    df_input = df_output.iloc[::ratio].copy()
    df_input.index = df_output.index[::ratio]

    return df_input, df_output


class DatasetGenerator(Dataset):
    """
    PyTorch Dataset class for generating time series training data with different
    sampling frequencies and interpolation methods
    """

    def __init__(self, df_input, df_output, input_length, output_length, s_in, s_out,
                 use_window, overlap_ratio=0.0, minmax_scaler=(-1, 1), extra_interpolate=False):
        """
        Initialize the preprocessed generator

        Args:
            df_input (pd.DataFrame): Input time series dataframe
            df_output (pd.DataFrame): Output time series dataframe (can be None)
            input_length (int): Length of input sequences
            output_length (int): Length of output sequences
            s_in (str): Input sampling frequency
            s_out (str): Output sampling frequency
            use_window (bool): Whether to use sliding window approach
            overlap_ratio (float): Overlap ratio between consecutive windows
            minmax_scaler (tuple): Min-max scaling range
            extra_interpolate (bool): Whether to generate additional interpolation variants
        """

        # === Automatically add 'date' column if missing ===
        if 'date' not in df_input.columns:
            df_input['date'] = df_input.index.date
        if df_output is not None and 'date' not in df_output.columns:
            df_output['date'] = df_output.index.date

        # Store configuration parameters
        self.df_input = df_input.copy()
        self.df_output = df_output.copy() if df_output is not None else df_input.resample(s_out).first().ffill()
        self.df_output_is_none = df_output is None
        self.input_length = input_length
        self.output_length = output_length
        self.s_in = s_in
        self.s_out = s_out
        self.use_window = use_window
        self.overlap_ratio = overlap_ratio
        self.mask = build_causal_mask(output_length, input_length)
        self.minmax_scaler = minmax_scaler


        # Initialize data storage lists
        self.x_input_list = []  # Original input sequences
        self.x_initial_list = []  # Processed initial sequences (interpolated/downsampled)
        self.x_initial_mask_list = []  # Masks indicating which positions contain original data
        self.x_output_list = []  # Target output sequences
        self.scaler_params_list = []

        # Extra interpolation methods (optional)
        self.extra_interpolate = extra_interpolate
        self.x_slinear_list = []  # Linear spline interpolation
        self.x_cubic_list = []  # Cubic interpolation
        self.x_quad_list = []  # Quadratic interpolation
        self.x_gp_list = []  # Gaussian Process interpolation

        # Generate windowed sequences if use_window is True
        if use_window:
            total_len = len(self.df_output)
            # Calculate step size based on overlap ratio
            step = max(1, int(self.output_length * (1 - self.overlap_ratio)))
            # Calculate sampling ratio between input and output frequencies
            ratio = int(pd.Timedelta(self.s_in) / pd.Timedelta(self.s_out))

            # Generate sliding windows
            for start in range(0, total_len - self.output_length + 1, step):
                # Extract output window
                window_output = self.df_output.iloc[start:start + self.output_length]
                if len(window_output) != self.output_length:
                    continue

                # Extract corresponding input window by downsampling
                window_input = window_output.iloc[::ratio].copy()
                if len(window_input) != self.input_length:
                    continue

                # Process this window pair
                self._process_window(window_input, window_output)

        # Convert lists to numpy arrays for efficient access
        self.x_input_list = np.array(self.x_input_list, dtype=np.float32)
        self.x_initial_list = np.array(self.x_initial_list, dtype=np.float32)
        self.x_output_list = np.array(self.x_output_list, dtype=np.float32)
        self.x_initial_mask_list = np.array(self.x_initial_mask_list, dtype=np.float32)
        self.x_slinear_list = np.array(self.x_slinear_list, dtype=np.float32)
        self.x_cubic_list = np.array(self.x_cubic_list, dtype=np.float32)
        self.x_quad_list = np.array(self.x_quad_list, dtype=np.float32)
        self.x_gp_list = np.array(self.x_gp_list, dtype=np.float32)
        self.scaler_params_list = np.array(self.scaler_params_list, dtype=np.float32)
        # print(f"interpolate done")

    def _process_window(self, df_in, df_out):
        """
        Process a single window pair of input and output dataframes

        Args:
            df_in (pd.DataFrame): Input window dataframe
            df_out (pd.DataFrame): Output window dataframe
        """
        # Select only numeric columns
        df_in = df_in.select_dtypes(include=['number'])
        df_out = df_out.select_dtypes(include=['number'])

        # Extract numpy arrays
        x_input = df_in.values
        if df_out is not None:
            x_output = df_out.values
            # Normalize input and output together for consistency
            x_input, x_output, scaler = normalize_pair(x_input, x_output, self.minmax_scaler)
        else:
            x_output = None
            scaler = MinMaxScaler(feature_range=self.minmax_scaler)
            x_input = scaler.fit_transform(x_input)

        scaler_params = np.stack([scaler.min_, scaler.scale_], axis=0)
        self.scaler_params_list.append(scaler_params)

        # Handle sequence length differences
        if self.output_length > self.input_length:
            # Upsample: interpolate from input to output length
            x_initial = interpolate_sequence(x_input, self.input_length, self.output_length, kind='linear')

            # Create mask indicating original data positions
            x_initial_mask = np.full((self.output_length, x_input.shape[1]), float('-inf'))
            for i in range(self.input_length):
                # Calculate position of original point in new sequence
                pos = int(i * (self.output_length - 1) / (self.input_length - 1))
                x_initial_mask[pos, :] = 1

            # Generate additional interpolation methods if requested
            if self.extra_interpolate:
                x_slinear = interpolate_sequence(x_input, self.input_length, self.output_length, kind='slinear')
                x_cubic = interpolate_sequence(x_input, self.input_length, self.output_length, kind='cubic')
                x_quad = interpolate_sequence(x_input, self.input_length, self.output_length, kind='quadratic')
                x_gp = interpolate_sequence_gp(x_input, self.input_length, self.output_length)
                self.x_slinear_list.append(x_slinear)
                self.x_cubic_list.append(x_cubic)
                self.x_quad_list.append(x_quad)
                self.x_gp_list.append(x_gp)

        elif self.output_length < self.input_length:
            # Downsample: average input values to match output length
            x_initial = downsample_sequence(x_input, self.input_length, self.output_length)
            x_initial_mask = np.full((self.output_length, x_input.shape[1]), float('-inf'))
        else:
            # Same length: use input as-is
            x_initial = x_input.copy()
            x_initial_mask = np.ones_like(x_initial)

        # Store processed data
        self.x_input_list.append(x_input)
        self.x_initial_list.append(x_initial)
        self.x_initial_mask_list.append(x_initial_mask)

        # Store output data or create dummy data if output is None
        if self.df_output_is_none is not None:
            self.x_output_list.append(x_output)
        else:
            self.x_output_list.append(np.zeros_like(x_initial))

    def __len__(self):
        """Return the number of samples in the preprocessed"""
        return len(self.x_input_list)

    def __getitem__(self, idx):
        """
        Get a single sample from the preprocessed

        Args:
            idx (int): Sample index

        Returns:
            tuple: (x_input, x_initial, x_initial_mask, x_output, mask, freqs, [interpolations])
        """
        # Convert to PyTorch tensors
        x_input = torch.FloatTensor(self.x_input_list[idx])
        x_initial = torch.FloatTensor(self.x_initial_list[idx])
        x_initial_mask = torch.FloatTensor(self.x_initial_mask_list[idx])
        x_output = torch.FloatTensor(self.x_output_list[idx]) if self.df_output is not None else None
        scaler_params = torch.FloatTensor(self.scaler_params_list[idx])

        # Return with or without extra interpolations
        if self.extra_interpolate:
            x_slinear = torch.FloatTensor(self.x_slinear_list[idx])
            x_cubic = torch.FloatTensor(self.x_cubic_list[idx])
            x_quad = torch.FloatTensor(self.x_quad_list[idx])
            x_gp = torch.FloatTensor(self.x_gp_list[idx])
            return x_input, x_initial, x_initial_mask, x_output, self.mask, [self.s_in, self.s_out, scaler_params], [x_slinear,
                                                                                                      x_cubic, x_quad, x_gp]

        return x_input, x_initial, x_initial_mask, x_output, self.mask, [self.s_in, self.s_out, scaler_params]

    @staticmethod
    def split_dataset(dataset, train_ratio=0.6, test_ratio=0.1, valid_ratio=0.3, random_seed=42, shuffle=True):
        """
        Split preprocessed into train, test, and validation sets

        Args:
            dataset (Dataset): Dataset to split
            train_ratio (float): Fraction for training set
            test_ratio (float): Fraction for test set
            valid_ratio (float): Fraction for validation set
            random_seed (int): Random seed for reproducibility
            shuffle (bool): Whether to shuffle data before splitting

        Returns:
            tuple: (train_dataset, test_dataset, valid_dataset)
        """
        # Verify ratios sum to 1
        assert abs(train_ratio + test_ratio + valid_ratio - 1.0) < 1e-5

        total_size = len(dataset)
        indices = list(range(total_size))

        if shuffle:
            # Use local random number generator for reproducibility
            rng = np.random.RandomState(random_seed)
            rng.shuffle(indices)

        # Calculate split points
        train_end = int(train_ratio * total_size)
        test_end = train_end + int(test_ratio * total_size)

        # Create subset datasets
        return (
            Subset(dataset, indices[:train_end]),  # Training set
            Subset(dataset, indices[train_end:test_end]),  # Test set
            Subset(dataset, indices[test_end:])  # Validation set
        )


if __name__ == '__main__':
    # === Step 1: Load pre-generated square wave data ===
    df = pd.read_csv("../../dataset/special_wave_square.csv")
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)
    df = df[["value"]]  # Ensure only value column is selected

    # === Step 2: Create aligned input/output datasets ===
    df_input, df_output = get_aligned_input_output(df, s_in="15min", s_out="5min")

    # === Step 3: Build preprocessed ===
    dataset = DatasetGenerator(
        df_input=df_input,
        df_output=df_output,
        input_length=97,  # Number of input time steps
        output_length=289,  # Number of output time steps
        s_in="15min",  # Input sampling frequency
        s_out="5min",  # Output sampling frequency
        use_window=True,  # Use sliding window approach
        overlap_ratio=0.3,  # 30% overlap between consecutive windows
    )

    # === Step 4: Split into train/test/validation sets ===
    train_dataset, test_dataset, valid_dataset = DatasetGenerator.split_dataset(dataset)

    # === Step 5: Examine a sample ===
    x_input, x_initial, x_initial_mask, x_output, mask, freqs = dataset[1]

    # Print sample information
    print(f"x_input shape: {x_input.shape}")
    print(f"x_initial shape: {x_initial.shape}")
    print(f"x_initial_mask shape: {x_initial_mask.shape}")
    print(f"x_output shape: {x_output.shape if x_output is not None else 'None'}")
    print(f"Causal mask shape: {mask.shape}")

    print(f"x_input data:\n{x_input.reshape(-1)}")
    print(f"x_initial data:\n{x_initial.reshape(-1)}")
    print(f"x_initial_mask data:\n{x_initial_mask.reshape(-1)}")
    print(f"x_output data:\n{x_output.reshape(-1) if x_output is not None else 'None'}")
    print(f"Mask data:\n{mask}")