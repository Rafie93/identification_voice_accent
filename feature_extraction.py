import os
import numpy as np
from scipy.io.wavfile import read
import matplotlib.pyplot as plt
import pandas as pd
from scipy.fftpack import dct
from python_speech_features import mfcc as melfcc
from matplotlib import cm

def mfcc(sample_rate,audio):
    signal = audio[0:int(3.5 * sample_rate)]
    # plt.subplot(321)
    # plt.plot(signal)
    # plt.xlabel('Time')
    # plt.ylabel('Amplitude')
    # plt.title('Signal in the Time Domain')

    # STEP 1 PRE EMPHASIS
    pre_emphasis = 0.97
    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

    # STEP 2 FRAMMING
    frame_size = 0.025
    frame_stride = 0.01

    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate
    signal_length = len(emphasized_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(
        np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(emphasized_signal,z)

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(
        np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]

    # STEP 3 WINDOWING
    frames *= np.hamming(frame_length)

    # STEP 4 FOURIER TRANSFORM FFT
    NFFT = 512
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum

    # STEP 5 MEL FILTERBANKS
    low_freq_mel = 0
    nfilt = 40
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10 ** (mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = np.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])  # left
        f_m = int(bin[m])  # center
        f_m_plus = int(bin[m + 1])  # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * np.log10(filter_banks)  # dB
    samplingFrequency = 400

    filter_banks -= (np.mean(filter_banks, axis=0) + 1e-8)
    # plt.imshow(filter_banks.T, cmap=plt.cm.jet, aspect='auto')
    # plt.xticks(np.arange(0, (filter_banks.T).shape[1],
    #                      int((filter_banks.T).shape[1] / 4)),
    #            ['0s', '0.5s', '1s', '1.5s', '2.5s', '3s', '3.5'])
    # ax = plt.gca()
    # ax.invert_yaxis()
    # plt.xlabel('Time')
    # plt.ylabel('Frequency')
    # plt.title('Spectrogram of the Signal')

    # STEP 6 MFCC
    num_ceps = 20
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1: (num_ceps + 1)]  #  20
    (nframes, ncoeff) = mfcc.shape
    n = np.arange(ncoeff)
    cep_lifter = 22
    lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
    mfcc *= lift
    mfcc_ori = mfcc

    plt.subplot(325)
    mfcc -= (np.mean(mfcc, axis=0) + 1e-8)
    # plt.imshow(mfcc.T, cmap=plt.cm.jet, aspect='auto')
    # plt.xticks(np.arange(0, (mfcc.T).shape[1],
    #                      int((mfcc.T).shape[1] / 4)),
    #            ['0s', '0.5s', '1s', '1.5s', '2.5s', '3s', '3.5'])
    # ax = plt.gca()
    # ax.invert_yaxis()
    # plt.xlabel('Time')
    # plt.ylabel('MFCC Coefficients')
    # plt.title('MFCC')

    # rataMFCC = mfcc.mean(axis=0)
    # print(rataMFCC)
    #NA = np.asarray(mfcc)
    NA = mfcc.astype(float)
    AVG = np.mean(NA, axis=0)
    #MN = np.min(NA, axis=0)
    #MX = np.max(NA, axis=0)

    hasil = (AVG)
    return hasil
#
# #path to training data
# source   = "D:/KULIAH/Digital Signal Processing/Identification Logat/data_training/"
# #path to save trained model
# dest     = "D:/KULIAH/Digital Signal Processing/Identification Logat/data_training/"
# files    = [os.path.join(source,f) for f in os.listdir(source) if
#              f.endswith('.wav')]
# features = np.asarray(());
#
# for f in files:
#     sr,audio = read(f)
#     vector   = mfcc(sr,audio)
#     if features.size == 0:
#         features = vector
#     else:
#         features = np.vstack((features, vector))
# np.savetxt("D:/KULIAH/Digital Signal Processing/pygender/dataset.csv",features)
#
