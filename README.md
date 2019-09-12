# Automatic baseline correction/estimation algorithm

This contains an algorithm to automatically estimate continua or broad underlying signals found in laser spectroscopy.

See references:

[1] Schulz et al (2011) "A Model-Free, Fully Automated Baseline-Removal Method for 
Raman Spectra"
[2] Schulz et al. (2012) "A Small-Window Moving Average-Based Fully Automated Baseline 
Estimation Method for Raman Spectra"

# Usage

The function requires `X` and `Y` data and a moving filter (Savitzky-Golay) span, `window`. 
An option is included to "fill" the recess created by a notch filter by setting `notch = True` and the wavelengths to fill between, `fill_between = [wvl1, wvl2]`. This returns the the estimated baseline only.

~~~py
baseline = estimateBaseline(X, Y, # X and Y data.
                    window, # Window span for the SG filter - must be odd number.
                    notch, # This linearly interpolates between two points if a notch filter is used (default = False).
                    fill_between, # Set the wavelengths within which to fill.
                    )
~~~

# Example

The examples folder contains a script `fit_libs_spectra.py` showing it's use on a laser-induced plasma signal. (In the main folder directory run: `python -m examples.fit_libs_spectra`)

![libs_baseline](/examples/libs_baseline.png)