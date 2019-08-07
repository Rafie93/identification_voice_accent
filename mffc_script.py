import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io.wavfile
from scipy.fftpack import dct
from python_speech_features import mfcc as melfcc
from matplotlib import cm

#STEP 0
filename='D:/KULIAH/Digital Signal Processing/Identification Logat/fahmi_ban_ngaju_(12).wav'
sample_rate, signal = scipy.io.wavfile.read(filename)
print(sample_rate)
signal = signal[0:int(3.5 * sample_rate)]
# plt.subplot(221)
# plt.plot(signal)
# plt.xlabel('Time')
# plt.ylabel('Amplitude')
# plt.title('Signal dalam Time Domain')

#STEP 1 PRE EMPHASIS
pre_emphasis = 0.97
emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
# plt.subplot(221)
# plt.plot(emphasized_signal)
# plt.xlabel('Time')
# plt.ylabel('Amplitude')
# plt.title('Signal dalam Time Domain Sesudah Pre-Emphasis')

#STEP 2 FRAMMING
frame_size = 0.025
frame_stride = 0.01

frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
signal_length = len(emphasized_signal)
frame_length = int(round(frame_length))
frame_step = int(round(frame_step))
num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

pad_signal_length = num_frames * frame_step + frame_length
z = np.zeros((pad_signal_length - signal_length))
pad_signal = np.append(emphasized_signal, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
frames = pad_signal[indices.astype(np.int32, copy=False)]
framming_plt = pad_signal[indices.astype(np.int32, copy=False)]

# plt.subplot(222)
plt.figure(figsize=(10, 4))
plt.plot(framming_plt)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Framing')
plt.show()
#STEP 3 WINDOWING
frames *= np.hamming(frame_length)

plt.figure(figsize=(10, 4))
plt.plot(frames)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Windowing')
plt.show()

#STEP 4 FOURIER TRANSFORM FFT
NFFT = 512
mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum
plt.figure(figsize=(10, 4))
plt.plot(pow_frames)
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.title('Fast Fourier Transform')
plt.show()
# plt.subplot(224)
# plt.plot(pow_frames)
# plt.xlabel('Frequency')
# plt.ylabel('Amplitude')
# plt.title('Fast Fourier Transform')

#STEP 5 MEL FILTERBANKS
low_freq_mel = 0
nfilt=40
high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
bin = np.floor((NFFT + 1) * hz_points / sample_rate)

fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
for m in range(1, nfilt + 1):
    f_m_minus = int(bin[m - 1])   # left
    f_m = int(bin[m])             # center
    f_m_plus = int(bin[m + 1])    # right

    for k in range(f_m_minus, f_m):
        fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
    for k in range(f_m, f_m_plus):
        fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

filter_banks = np.dot(pow_frames, fbank.T)
filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
filter_banks = 20 * np.log10(filter_banks)  # dB
# plt.subplot(222)
plt.figure(figsize=(10, 4))
plt.plot(filter_banks)
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.title('Filter Bank Mel Scale')
plt.show()

#plt.subplot(223)
filter_banks -= (np.mean(filter_banks,axis=0) + 1e-8)
plt.imshow(filter_banks.T, cmap=plt.cm.jet, aspect='auto')
plt.xticks(np.arange(0, (filter_banks.T).shape[1],
int((filter_banks.T).shape[1] / 4)),
['0s', '0.5s', '1s', '1.5s','2.5s','3s','3.5'])
ax = plt.gca()
ax.invert_yaxis()
# plt.xlabel('Time')
# plt.ylabel('Frequency')
# plt.title('Spectrogram of the Signal')


#STEP 6 MFCC
num_ceps = 21
mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)] # Keep 2-13
(nframes, ncoeff) = mfcc.shape
n = np.arange(ncoeff)
cep_lifter = 22
lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
mfcc *= lift
mfcc_ori = mfcc

# plt.subplot(224)
mfcc -= (np.mean(mfcc,axis=0) + 1e-8)
plt.imshow(mfcc.T, cmap=plt.cm.jet, aspect='auto')
plt.xticks(np.arange(0, (mfcc.T).shape[1],
int((mfcc.T).shape[1] / 4)),
['0s', '0.5s', '1s', '1.5s','2.5s','3s','3.5'])
ax = plt.gca()
ax.invert_yaxis()
plt.xlabel('Time')
plt.ylabel('MFCC Coefficients')
plt.title('MFCC')
plt.show()

#rataMFCC = mfcc.mean(axis=0)
#print(rataMFCC)
NA = np.asarray(mfcc)
NA = mfcc.astype(float)
AVG = np.mean(NA, axis=0)
MN = np.min(NA, axis=0)
MX = np.max(NA, axis=0)

hasil=(AVG, MN, MX)

np.savetxt("D:/KULIAH/Digital Signal Processing/rata.csv",hasil,
           delimiter=",",header="MFCC1,MFCC2,MFCC3,MFCC4,MFCC5,MFCC6,MFCC7,MFCC8,MFCC9,MFCC10,MFCC11,MFCC12,MFCC13,"
                                "MFCC14,MFCC15,MFCC16,MFCC17,MFCC18,MFCC19,MFCC20,LABEL")


#print(mfcc)
# plt.show()

#
# (rate,sig) = scipy.io.wavfile.read(filename)
# mfcc_feat = melfcc(sig,rate)
# ig, ax = plt.subplots()
# mfcc_data= np.swapaxes(mfcc_feat, 0 ,1)
# cax = ax.imshow(mfcc_data, interpolation='nearest', cmap=cm.coolwarm, origin='lower', aspect='auto')
# ax.set_title('MFCC')
# #Showing mfcc_data
# plt.show()
# #Showing mfcc_feat
# plt.plot(mfcc_feat)
# plt.show()
# print(mfcc_data)
#
# rataMFCC2 = mfcc_data.mean(axis=0)
# print(rataMFCC2)
# np.savetxt("D:/KULIAH/Digital Signal Processing/rata2.csv", rataMFCC2, delimiter=",")
