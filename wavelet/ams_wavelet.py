import pandas as pd
import numpy as np
from waveletFunctions import wave_signif, wavelet


# Thanks to Evgeniya Predybaylo for the Python wavelet code
__author__ = 'Evgeniya Predybaylo'

# See "http://paos.colorado.edu/research/wavelets/"
# The Matlab code written January 1998 by C. Torrence
# modified to Python by Evgeniya Predybaylo, December 2014

# Python wavelet software provided by Evgeniya Predybaylo based on Torrence and Compo (1998) 
# and is available at URL: "http://atoc.colorado.edu/research/wavelets/"


def calculate_alpha(time_series):
    time_series = np.array(time_series)
    N = len(time_series)
    mean_x = np.mean(time_series)
    
    numerator = sum((time_series[n] - mean_x) * (time_series[n+1] - mean_x) for n in range(N-1))
    denominator = sum((time_series[n] - mean_x)**2 for n in range(N))
    
    alpha = numerator / denominator
    return alpha


def waveletAnalysis(flux_0):

    N = flux_0.size
    dt = 1  # In days

    p = np.polyfit(np.arange(N), flux_0, 6)
    # flux_notrend = flux_0 - np.polyval(p, np.arange(N))

    flux_notrend = flux_0 - np.mean(flux_0)
    # flux = flux_notrend
    flux = (flux_notrend)/np.std(flux_notrend, ddof=1)

    variance = np.var(flux, ddof=1) # should be 1
    print("variance = ", variance)

    # xlim = [date_begin, date_end]
    pad = 1  # pad the time series with zeroes (recommended)
    dj = 1/32  # this will do * sub-octaves per octave
    s0 = 4 * dt  # this says start at a scale of * days
    j1 = 128 # this says do * sub-octaves each
    lag1 = calculate_alpha(flux)
    print("lag1 = ", lag1)

    mother = 'MORLET'

    # Wavelet transform:
    wave, period, scale, coi = wavelet(flux, dt, pad, dj, s0, j1, mother)
    power = (np.abs(wave)) ** 2  # compute wavelet power spectrum
    global_ws = (np.sum(power, axis=1) / N)  # time-average over all times

    # Significance levels:
    signif = wave_signif(([variance]), dt=dt, sigtest=0, scale=scale,
        lag1=lag1, mother=mother)
    # expand signif --> (J+1)x(N) array
    sig95 = signif[:, np.newaxis].dot(np.ones(N)[np.newaxis, :])
    sig95 = power / sig95  # where ratio > 1, power is significant

    # Global wavelet spectrum & significance levels:
    dof = N - scale  # the -scale corrects for padding at edges
    global_signif = wave_signif(variance, dt=dt, scale=scale, sigtest=1,
        lag1=lag1, dof=dof, mother=mother)


    return flux_0, period, power, coi, sig95, global_ws, global_signif