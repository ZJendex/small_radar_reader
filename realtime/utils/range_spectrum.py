import numpy as np
RANGE_FFT = 512
AZIM_FFT = 64
MAX_RANGE = 21.6

def compute_range_spectrum_onChirp(signal):# signal shape (num_chirp_samples,)
    Y = np.fft.fft(signal, 512)
    Y= np.squeeze(Y)
    Y_power_profile = np.abs(Y)
    range_meter = 0.0422 * np.arange(0, 512)
    return range_meter, Y_power_profile


def compute_rangeFFT_BF(signal): # signal shape (num_Tx, num_Rx, num_chirp_samples)

    fft_array = np.concatenate((signal[0,:,:], signal[2,:,:]),axis=0)
    first_fft = np.fft.fft(fft_array, n=RANGE_FFT, axis=1)
    second_fft = np.fft.fft(first_fft, n=AZIM_FFT, axis=0)
    second_fft = np.fft.fftshift(second_fft, axes=0)
    second_fft = np.abs(second_fft)
    return second_fft

def construct_polar_space():
    sine_theta = -2*np.linspace(-0.5,0.5,AZIM_FFT)
    cos_theta = np.sqrt(1-sine_theta**2)
    range_d, sine_theta_mat = np.meshgrid(np.linspace(0,MAX_RANGE,RANGE_FFT),sine_theta)
    _, cos_theta_mat = np.meshgrid(np.linspace(0,MAX_RANGE,RANGE_FFT),cos_theta)
    x_axis = np.multiply(range_d, cos_theta_mat)
    y_axis = np.multiply(range_d, sine_theta_mat)
    return x_axis, y_axis