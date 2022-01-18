# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 12:37:33 2021

@author: Mark
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

f_name = "9_fret_pluck1_quick"

file = "C:\\Users\\mbeye\\OneDrive\\3117Lab2\\" + f_name + ".txt"
f = open(file)
data = f.readlines()

time = np.array([float(data[i+2].split()[0]) for i in range(len(data)-2)])
voltage = np.array([float(data[i+2].split()[1]) for i in range(len(data)-2)])

time_sub = time[0:len(time)]
voltage_sub = voltage[0:len(voltage)]

plt.plot(time_sub, voltage_sub)

#%%

dt = time[1]

def fourier_spectrum(spacing,channel,kaiser=False):
#Simple function to automate the fft process. Time-spacing between data points, array of y-data to be transformed. 
#kaiser=True appends a Kaiser window to the data.
    number_samples = len(channel)
    if kaiser==True:
        channel = channel*np.kaiser(number_samples,2)
    
    freq_space = np.fft.rfftfreq(number_samples,spacing)
    fft_space = np.fft.rfft(channel)*2/number_samples
    fft_x = freq_space[:number_samples//2]
    fft_y = abs(fft_space[:number_samples//2])
    return fft_x,fft_y #Frequency x-axis data, fourier y-axis data.

fft_freq, fft_amp = fourier_spectrum(dt, voltage_sub-np.mean(voltage_sub))

peaks = find_peaks(fft_amp, height = (0.005))
height = peaks[1]['peak_heights'] #list of the heights of the peaks
peak_pos = fft_freq[peaks[0]] #list of the peaks positions

fig = plt.figure(figsize=(8,3), dpi=200)
ax = fig.subplots()
ax.plot(fft_freq, fft_amp)
ax.scatter(peak_pos, height, color = 'r', s = 15, marker = 'D', label = 'Maxima')
ax.grid()
ax.set_title("Fourier power spectral density ")
ax.set_ylabel("Power spectral density")
ax.set_xlabel("Frequency (Hz)")
plt.yscale('log')
ax.set_xlim(0, 1000)
ax.set_ylim(10**(-5), 1)

#%%
x_m = 14.8
L = 64.5
pi = 3.14159
harmonics = [[np.absolute(np.round((height[i]/height[0])*np.sin(x_m*pi/L)/(i*np.sin(i*x_m*pi/L)), 3)) for i in range(1,7)]]

# test = np.empty((0,6), float)

test = np.append(test, np.array(harmonics), axis=0)
#%%

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def fourier_spectrum(spacing,channel,kaiser=False):
#Simple function to automate the fft process. Time-spacing between data points, array of y-data to be transformed. 
#kaiser=True appends a Kaiser window to the data.
    number_samples = len(channel)
    if kaiser==True:
        channel = channel*np.kaiser(number_samples,2)
    
    freq_space = np.fft.rfftfreq(number_samples,spacing)
    fft_space = np.fft.rfft(channel)*2/number_samples
    fft_x = freq_space[:number_samples//2]
    fft_y = abs(fft_space[:number_samples//2])
    return fft_x,fft_y #Frequency x-axis data, fourier y-axis data.

fret_list = [str(i) + "_fret_pluck1_quick" for i in range(1, 13)]

harmonic_array = [0]

for x in fret_list:
    file = "C:\\Users\\mbeye\\OneDrive\\3117Lab2\\" + x + ".txt"
    f = open(file)
    data = f.readlines()
    
    time = np.array([float(data[i+2].split()[0]) for i in range(len(data)-2)])
    voltage = np.array([float(data[i+2].split()[1]) for i in range(len(data)-2)])
    
    dt = time[1]
    
    fft_freq, fft_amp = fourier_spectrum(dt, voltage-np.mean(voltage))

    peaks = find_peaks(fft_amp, height = (0.0025))
    height = peaks[1]['peak_heights'] #list of the heights of the peaks
    peak_pos = fft_freq[peaks[0]] #list of the peaks positions
    
    np.append(harmonic_array, height, axis=0)
    
    
