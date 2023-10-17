import numpy as np

# Function to calculate variance of logits
def calculate_variance(logits):
    # Calculate absolute values of logits
    abs_logits = np.abs(logits)
    # Calculate variance of absolute logits
    variance = np.var(abs_logits)
    return variance

# Function to detect unknown samples
def detect_unknown(variance, threshold):
    if variance < threshold:
        return True  # Input is an unknown sample
    else:
        return False  # Input is a known sample