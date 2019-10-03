"""
Automatic Baseline Fitting Tool. Modified from MATLAB script. [APW-UB + UNK 2019]

Applies automatic baseline estimation according to:

[1] Schulz et al (2011) "A Model-Free, Fully Automated Baseline-Removal Method for 
Raman Spectra"
[2] Schulz et al. (2012) "A Small-Window Moving Average-Based Fully Automated Baseline 
Estimation Method for Raman Spectra"
"""

import math
import numpy as np
from scipy import signal

def Noise(Y): 

    """
    The function calculates the standard deviation from the baseline (the noise). For this,
    the spectrum is shifted two times to the right, one pixel at a time, and subtracted from
    each other. That only root 6 times the noise remains. Larger deviations than the
    three-times noise value are replaced and then the final value is calculated.
    """


    a1 = np.pad(Y, (0, 2), 'constant', constant_values=(0, 0))
    a2 = np.pad(Y, (1, 1), 'constant', constant_values=(0, 0))
    a3 = np.pad(Y, (2, 0), 'constant', constant_values=(0, 0))

    G = ((a3 - a2) - (a2 - a1))

    noise = np.std(G[2:])
     
    G[:] = [noise * 3 if j > (noise * 3) else -(noise * 3) if j < -(noise * 3) else j for j in G]

    noise = np.sqrt(np.var(G[2:])/6)

    return noise 

def fillNotch(X, Y, fill_between):

    """
    Fills notch created by interference or absorption filter for better baseline 
    fitting. Linearly extrapolates between two selected wavelengths.
    """

    lower_idx = np.searchsorted(X, fill_between[0], side="left")
    upper_idx = np.searchsorted(X, fill_between[1], side="right")

    gradient = (Y[upper_idx] - Y[lower_idx])/(X[upper_idx] - X[lower_idx])
    intersect = Y[upper_idx] - gradient * X[upper_idx]

    Y_filled = [gradient * i + intersect for i in X[lower_idx:upper_idx]]

    Y[lower_idx:upper_idx] = Y_filled

    return Y


def estimateBaseline(X, Y, window, notch = False, fill_between = None):

    """
    The function "estimateBaseline" is able to calculate both the offset of the baseline 
    of a spectrum and the curvature of a curved baseline. For this, however, the 
    window size `window` must be selected at the beginning of the correction. For
    spectra with superimposed peaks, where the superposition is to be obtained, must
    be started with the largest possible output window (1599 pixels). For strongly
    curved baselines should be started with much smaller window size (about 161 pixels)
     
    `noise` is the noise value of the baseline.
    `array_len` is the total number of pixels of the recorded spectrum.

    Using the Savitzky-Golay filter, the average of the pixels in the window size is 
    calculated and this value is transferred to the middle pixel. In the subsequent for 
    loop, the peaks are removed from the output spectrum, so the output spectrum is 
    peak-stripped.

    By means of the Chi ^ 2 criterion, it is tested how large the deviation between a 
    straight line (zero vector) and the difference between original stripped spectrum and 
    baseline is, if this is greater than or equal to the total number of pixels, then in 
    the while loop the Baseline calculated.

    It is tested via chi2 whether the deviation between the original spectrum and the 
    baseline corrected spectrum to which the mean of the baseline is added in order to 
    equalize the two spectra becomes too large. If the deviation becomes too large, the 
    window size is reduced in the if loop and made an odd number with the second if loop.
    """

    if not window % 2 == 1:
         raise ValueError("Window span must be an odd integer")

    if notch:
                       
        if len(fill_between) != 2:
            raise ValueError("'fill_between' wavelengths not correctly (e.g. fill_between = [lower, upper])")           

        if not np.min(X) < fill_between[0] < np.max(X):
            raise ValueError("'fill_between' lower bound out of range")
        
        if not np.min(X) < fill_between[1] < np.max(X):
            raise ValueError("'fill_between' upper bound out of range")

        Y = fillNotch(X, Y, fill_between)

    array_len = len(X)
    empty_array = np.zeros(array_len)   
    length_to_window = math.floor(array_len/window)
    
    noise = Noise(Y)

    baseline = signal.savgol_filter(Y, window, 0)

    bl_in_progress = np.array(baseline)

    for i in range(array_len):

        if Y[i] > bl_in_progress[i] * 2 * noise:

            bl_in_progress[i] = bl_in_progress[i] + 1.99 * noise * np.random.randn()

        else:

            pass


    residuals = bl_in_progress - baseline

    chi = np.sum(((empty_array - residuals) ** 2) / (noise ** 2))

    j = 0

    smooth_orig_signal = np.zeros(array_len)

    while chi >= array_len:

        orig_signal = list(Y)

        #bl_in_progress[bl_in_progress > (baseline + 2 * noise)] = baseline + 1.99 * noise * np.random.randn()

        for i in range(array_len):

            if bl_in_progress[i] > baseline[i] + 2 * noise:

                bl_in_progress[i] = baseline[i] + 1.99 * noise * np.random.randn()

            else:

                pass


        Y_avg_left = np.mean(Y[0:window])
        Y_avg_right = np.mean(Y[window:-1])
        
        # Start pad-smooth-unpad

        bl_padded = np.pad(bl_in_progress, (window, window), 'constant', constant_values=(Y_avg_left, Y_avg_right))

        bl_smoothed = signal.savgol_filter(bl_padded, window, 0)

        baseline = bl_smoothed[window:-window]

        # End pad-smooth-unpad

        for idx in range(array_len):
            if orig_signal[idx] > baseline[idx] + 2 * noise:
                smooth_orig_signal[idx] = baseline[idx]
            else:
                smooth_orig_signal[idx] = orig_signal[idx]

        residuals = smooth_orig_signal - baseline
        test = Y - baseline
        chi = np.sum(((empty_array - residuals) ** 2) / noise ** 2)
        chi2 = np.sum(((Y - (test + np.mean(baseline))) ** 2) / noise ** 2)

        j += 1

        if chi2 > array_len:

            length_to_window += 1

            window = round(array_len/length_to_window)

            if window % 2 == 0:

                window += 1
            
            bl_in_progress = Y

            j = 0

    return baseline



