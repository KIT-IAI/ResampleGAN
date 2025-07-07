import numpy as np
from scipy import signal
import pandas as pd


# Define waveform generation functions

def generate_square_wave(length, freq=5, amplitude=1.0):
    """
    Generate a square wave signal

    Args:
        length (int): Number of data points to generate
        freq (float): Frequency of the wave (default: 5)
        amplitude (float): Amplitude of the wave (default: 1.0)

    Returns:
        numpy.ndarray: Square wave signal values
    """
    # Create time vector from -1 to 1
    t = np.linspace(-1, 1, length, endpoint=False)
    # Generate square wave using scipy.signal.square
    return amplitude * signal.square(2 * np.pi * freq * t)


def generate_sawtooth_wave(length, freq=5, amplitude=1.0):
    """
    Generate a sawtooth wave signal

    Args:
        length (int): Number of data points to generate
        freq (float): Frequency of the wave (default: 5)
        amplitude (float): Amplitude of the wave (default: 1.0)

    Returns:
        numpy.ndarray: Sawtooth wave signal values
    """
    # Create time vector from -1 to 1
    t = np.linspace(-1, 1, length, endpoint=False)
    # Generate sawtooth wave using scipy.signal.sawtooth
    return amplitude * signal.sawtooth(2 * np.pi * freq * t)


def generate_triangle_wave(length, freq=5, amplitude=1.0):
    """
    Generate a triangle wave signal

    Args:
        length (int): Number of data points to generate
        freq (float): Frequency of the wave (default: 5)
        amplitude (float): Amplitude of the wave (default: 1.0)

    Returns:
        numpy.ndarray: Triangle wave signal values
    """
    # Create time vector from -1 to 1
    t = np.linspace(-1, 1, length, endpoint=False)
    # Generate triangle wave using scipy.signal.sawtooth with 0.5 width parameter
    # The 0.5 width parameter creates a symmetric triangle wave
    return amplitude * signal.sawtooth(2 * np.pi * freq * t, 0.5)


def generate_straight_line(length, slope=0.5, intercept=0.0):
    """
    Generate a straight line signal

    Args:
        length (int): Number of data points to generate
        slope (float): Slope of the line (default: 0.5)
        intercept (float): Y-intercept of the line (default: 0.0)

    Returns:
        numpy.ndarray: Linear signal values
    """
    # Create time vector from -1 to 1
    t = np.linspace(-1, 1, length, endpoint=False)
    # Generate linear function: y = slope * x + intercept
    return slope * t + intercept


def generate_sine_wave(length, freq=5, amplitude=1.0):
    """
    Generate a sine wave signal

    Args:
        length (int): Number of data points to generate
        freq (float): Frequency of the wave (default: 5)
        amplitude (float): Amplitude of the wave (default: 1.0)

    Returns:
        numpy.ndarray: Sine wave signal values
    """
    # Create time vector from -1 to 1
    t = np.linspace(-1, 1, length, endpoint=False)
    # Generate sine wave using numpy.sin
    return amplitude * np.sin(2 * np.pi * freq * t)


# Generate preprocessed and save as CSV file
def create_special_wave_dataset(output_path, length, freq, amplitude, wave_type):
    """
    Create a time-series preprocessed with specified waveform and save to CSV

    Args:
        output_path (str): File path where the CSV will be saved
        length (int): Number of data points to generate
        freq (float): Frequency parameter for wave generation
        amplitude (float): Amplitude parameter for wave generation
        wave_type (str): Type of wave to generate ('square', 'sawtooth', 'triangle', 'sine', 'line')

    Raises:
        ValueError: If unsupported wave_type is provided
    """
    # Create time index starting from 2018-01-01 with 5-minute intervals
    time_index = pd.date_range(start="2018-01-01", periods=length, freq="5min")

    # Generate data based on the specified wave type
    if wave_type == "square":
        data = generate_square_wave(length, freq, amplitude)
    elif wave_type == "sawtooth":
        data = generate_sawtooth_wave(length, freq, amplitude)
    elif wave_type == "triangle":
        data = generate_triangle_wave(length, freq, amplitude)
    elif wave_type == "sine":
        data = generate_sine_wave(length, freq, amplitude)
    elif wave_type == "line":
        # For line type, use amplitude as slope parameter
        data = generate_straight_line(length, slope=amplitude, intercept=0.0)
    else:
        # Raise error for unsupported wave types
        raise ValueError("Unsupported wave type")

    # Create DataFrame with time and value columns
    df = pd.DataFrame({"time": time_index, "value": data})

    # Save DataFrame to CSV file without row indices
    df.to_csv(output_path, index=False)

    # Print confirmation message
    print(f"Dataset saved to {output_path}")