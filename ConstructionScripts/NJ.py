# -*- coding: utf-8 -*-
"""before use, you need to change the all file paths in this script"""
import argparse
import array
import math
import numpy
import numpy as np
import random
import wave


def cal_adjusted_rms(clean_rms, snr):

    a = float(snr) / 20

    noise_rms = clean_rms / (10**a)
    return noise_rms


def cal_amp(wf):

    buffer = wf.readframes(wf.getnframes())

    amptitude = (np.frombuffer(buffer, dtype="int16")).astype(np.float64)
    return amptitude


def cal_rms(amp):

    return np.sqrt(np.mean(np.square(amp), axis=-1))


def save_waveform(output_path, params, amp):
    output_file = wave.Wave_write(output_path)
    output_file.setparams(params)
    output_file.writeframes(array.array("h", amp.astype(np.int16)).tobytes())
    output_file.close()


import numpy as np
from audiomentations import AddBackgroundNoise, Compose
import soundfile as sf


# 加载您的原始音频
def noise_add(clean_file, noise_file, output_mixed_file, snr):
    audio, sample_rate = sf.read(clean_file)

    noise_files = [noise_file]

    augmenter = Compose(
        [
            AddBackgroundNoise(
                sounds_path=noise_files,
                min_snr_in_db=snr,
                max_snr_in_db=snr,
                # noise_transform=lambda audio: audio,
                p=1.0,
            )
        ]
    )

    augmented_audio = augmenter(samples=audio, sample_rate=sample_rate)

    sf.write(output_mixed_file, augmented_audio, sample_rate)


import os
import random


def noise_choice(noise_dir):
    wav_files = [
        os.path.join(root, file)
        for root, dirs, files in os.walk(noise_dir)
        for file in files
        if file.endswith(".wav")
    ]
    return random.choice(wav_files)


import numpy as np
import wave
from scipy.signal import resample
import pandas as pd


def resample_and_compare_duration(src_file, noise_file, resampled_noise_file):
    # 读取原音频文件和噪声文件的采样率和数据
    with wave.open(src_file, "r") as src_wav:
        src_rate = src_wav.getframerate()
        src_frames = src_wav.readframes(src_wav.getnframes())
        src_data = np.frombuffer(src_frames, dtype=np.int16)

    with wave.open(noise_file, "r") as noise_wav:
        noise_rate = noise_wav.getframerate()
        noise_frames = noise_wav.readframes(noise_wav.getnframes())
        noise_data = np.frombuffer(noise_frames, dtype=np.int16)

    # 对噪声进行重采样
    noise_data_resampled = resample(
        noise_data, len(noise_data) * src_rate // noise_rate
    )

    # 计算原音频文件和重采样后的噪声文件的时长
    src_duration = len(src_data) / src_rate
    noise_duration = len(noise_data_resampled) / src_rate
    with wave.open(resampled_noise_file, "w") as resampled_noise_wav:
        resampled_noise_wav.setnchannels(1)
        resampled_noise_wav.setsampwidth(2)
        resampled_noise_wav.setframerate(src_rate)
        resampled_noise_wav.writeframes(noise_data_resampled.astype(np.int16).tobytes())

    # 判断噪声时长是否长于原音频文件时长
    return noise_duration >= src_duration


import shutil


def convert_audio2wav_files(src_dir, dst_dir, noise_dir, resampled_noise_file, snr):
    dst_dir = dst_dir + str(snr)
    for root, dirs, files in os.walk(src_dir):

        for file in files:
            src_file = os.path.join(root, file)
            dst_file = os.path.join(dst_dir, os.path.relpath(src_file, src_dir))

            # 创建目标文件的目录
            os.makedirs(os.path.dirname(dst_file), exist_ok=True)

            # 根据文件扩展名进行不同的转换操作

            if file.endswith(".wav"):
                try:

                    noise_file = noise_choice(noise_dir)
                    # if resample_and_compare_duration(src_file, noise_file,resampled_noise_file):
                    # break

                    noise_add(
                        clean_file=src_file,
                        noise_file=noise_file,
                        output_mixed_file=dst_file,
                        snr=snr,
                    )
                    # else:
                    # noise_add2(clean_file=src_file, noise_file=resampled_noise_file, output_mixed_file=dst_file, snr=snr)
                except Exception as e:
                    print(f"Error: {e}")
                    if not os.path.exists("addnoiseerror.csv"):
                        df = pd.DataFrame(columns=["src_file", "dst_file", "error"])
                        df.to_csv("addnoiseerror.csv", index=False)
                    df = pd.read_csv("addnoiseerror.csv")
                    df.loc[len(df)] = [src_file, dst_file, e]
                    df.to_csv("addnoiseerror.csv", index=False)

            else:
                # 对于其他格式的文件，直接复制到目标目录
                shutil.copy2(src_file, dst_file)


dst_dir = [
    "NoiseAdded-animal",
    "NoiseAdded-human",
    "NoiseAdded-natural",
    "NoiseAdded-exterior",
    "NoiseAdded-interior",
]


"""you need to categorize ESC-50 according to this classification"""


noise_dir = [
    "ESC-50-master/animal",
    "ESC-50-master/human",
    "ESC-50-master/natural",
    "ESC-50-master/exterior",
    "ESC-50-master/interior",
]

for i in range(len(dst_dir)):
    convert_audio2wav_files(
        dst_dir=dst_dir[i],
        src_dir="Std.subset",
        noise_dir=noise_dir[i],
        resampled_noise_file="resampled_noise.wav",
        snr=15,
    )
    convert_audio2wav_files(
        dst_dir=dst_dir[i],
        src_dir="Std.subset",
        noise_dir=noise_dir[i],
        resampled_noise_file="resampled_noise.wav",
        snr=20,
    )
    convert_audio2wav_files(
        dst_dir=dst_dir[i],
        src_dir="Std.subset",
        noise_dir=noise_dir[i],
        resampled_noise_file="resampled_noise.wav",
        snr=25,
    )


"""gaussion nosie"""

import numpy as np
import soundfile as sf


def add_gauss_noise(src, dst, snr):

    audio_file = src
    data, samplerate = sf.read(audio_file)

    desired_snr_db = snr

    audio_rms = np.sqrt(np.mean(data**2))
    desired_noise_rms = audio_rms / (10 ** (desired_snr_db / 20))

    noise = np.random.randn(len(data)) * desired_noise_rms

    noisy_data = data + noise

    noisy_audio_file = dst
    sf.write(noisy_audio_file, noisy_data, samplerate)


def gen_gauss(src, dst, snr):
    for root, dir, files in os.walk(src):
        os.makedirs(root.replace(src, dst), exist_ok=True)
        for file in files:
            if file.endswith(".wav"):
                try:

                    add_gauss_noise(
                        os.path.join(root, file),
                        os.path.join(root.replace(src, dst), file),
                        snr,
                    )
                except Exception as e:
                    print(f"Error: {e}")
            else:
                shutil.copy(
                    os.path.join(root, file), os.path.join(root.replace(src, dst), file)
                )


src = "std.subset"
dst = ["gaussadded15", "gaussadded20", "gaussadded25"]
snr = [15, 20, 25]
for i in range(len(dst)):
    gen_gauss(src, dst[i], snr[i])
