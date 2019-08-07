import librosa
import librosa.feature as fitur
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import pathlib
import csv


import warnings
warnings.filterwarnings('ignore')

cmap = plt.get_cmap('inferno')

plt.figure(figsize=(10,10))
suku = 'banjar_hulu banjar_kuala dayak_bakumpai dayak_ngaju'.split()
for g in suku:
    pathlib.Path(f'data_training/{g}').mkdir(parents=True, exist_ok=True)
    for filename in os.listdir(f'data_training/{g}'):
        songname = fdata_training/{g}/{filename}'
        y, sr = librosa.load(songname, mono=True, duration=5)
        plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB')
        plt.axis('off')
        plt.savefig(f'img_data/{g}/{filename[:-3].replace(".", "")}.png')
        plt.clf()


header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth zero_crossing_rate'
for i in range(1, 21):
    header += f' mfcc{i}'
header += ' label'
header = header.split()

file = open('data_training.csv', 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)
sukus = 'banjar_hulu banjar_kuala dayak_bakumpai dayak_ngaju'.split()
for g in sukus:
    for filename in os.listdir(f'data_training/{g}'):
        songname = f'data_training/{g}/{filename}'
        y, sr = librosa.load(songname, mono=True, duration=30)
        chroma_stft = fitur.chroma_stft(y=y, sr=sr)
        spec_cent = fitur.spectral_centroid(y=y, sr=sr)
        spec_bw = fitur.spectral_bandwidth(y=y, sr=sr)
        rmse = fitur.rmse(y)
        zcr = fitur.zero_crossing_rate(y)
        mfcc = fitur.mfcc(y=y, sr=sr)
        to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)}  {np.mean(zcr)}'
        for e in mfcc:
            to_append += f' {np.mean(e)}'
        to_append += f' {g}'
        file = open('data_training.csv', 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())