import numpy as np

def compute_range_spectrum(signal):
    # signal shape (num_chirp_samples,)
    Y = np.fft.fft(signal, 512)
    Y= np.squeeze(Y)
    Y_power_profile = np.abs(Y)
    range_meter = 0.0422 * np.arange(0, 512)
    return range_meter, Y_power_profile