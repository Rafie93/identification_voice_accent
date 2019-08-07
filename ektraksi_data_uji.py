import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import pathlib
import csv


import warnings
warnings.filterwarnings('ignore')

header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth zero_crossing_rate'
for i in range(1, 21):
    header += f' mfcc{i}'
header += ' label'
header = header.split()

file = open('D:/KULIAH/Digital Signal Processing/Identification Logat/data_testing.csv', 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)
suku = 'banjar_hulu banjar_kuala dayak_bakumpai dayak_ngaju'.split()
for g in suku:
    for filename in os.listdir(f'D:/KULIAH/Digital Signal Processing/Identification Logat/data_testing/{g}'):
        songname = f'D:/KULIAH/Digital Signal Processing/Identification Logat/data_testing/{g}/{filename}'
        y, sr = librosa.load(songname, mono=True, duration=30)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
      #  rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        rmse = librosa.feature.rmse(y)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)}  {np.mean(zcr)}'
        for e in mfcc:
            to_append += f' {np.mean(e)}'
        to_append += f' {g}'
        file = open('D:/KULIAH/Digital Signal Processing/Identification Logat/data_testing.csv', 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())