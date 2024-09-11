from pydub import AudioSegment
import soundfile as sf
import librosa
import os
import shutil
import numpy as np
from scipy import signal
from tqdm import tqdm


def mp32wav(src, dst):
    for root, dir, files in os.walk(src):
        for file in files:
            if file.endswith(".mp3"):
                try:
                    audio = AudioSegment.from_mp3(os.path.join(root, file))
                    os.makedirs(root.replace(src, dst), exist_ok=True)
                    audio.export(
                        os.path.join(
                            root.replace(src, dst), file.replace(".mp3", ".wav")
                        ),
                        format="wav",
                    )
                except:
                    print(os.path.join(root, file))

            else:
                os.makedirs(root.replace(src, dst), exist_ok=True)
                shutil.copy(
                    os.path.join(root, file), os.path.join(root.replace(src, dst), file)
                )


def myresample(src, dst):
    for root, dir, files in os.walk(src):
        for file in files:
            if file.endswith(".wav"):
                # try:
                y, sr = librosa.load(os.path.join(root, file), sr=None)
                y_resample = librosa.resample(y, orig_sr=sr, target_sr=16000)
                os.makedirs(root.replace(src, dst), exist_ok=True)
                sf.write(os.path.join(root.replace(src, dst), file), y_resample, 16000)
            # except Exception as e:
            #     print(os.path.join(root, file))
            #     print(e)
            else:
                os.makedirs(root.replace(src, dst), exist_ok=True)
                shutil.copy(
                    os.path.join(root, file), os.path.join(root.replace(src, dst), file)
                )


def convertMono(src, dst):
    for root, dir, files in os.walk(src):
        for file in files:
            if file.endswith(".wav"):
                try:
                    y, sr = librosa.load(os.path.join(root, file), sr=None)
                    if y.ndim >= 1:
                        y_mono = librosa.to_mono(y)
                        os.makedirs(root.replace(src, dst), exist_ok=True)
                        sf.write(os.path.join(root.replace(src, dst), file), y_mono, sr)
                except:
                    print(os.path.join(root, file))
            else:
                os.makedirs(root.replace(src, dst), exist_ok=True)
                shutil.copy(
                    os.path.join(root, file), os.path.join(root.replace(src, dst), file)
                )


def rmSilence(src, dst):

    for root, dir, files in os.walk(src):
        for file in files:
            if file.endswith(".wav"):
                try:
                    audio_file = os.path.join(root, file)
                    y, sr = librosa.load(audio_file)
                    # Audio(y, rate=sr)
                    # display(Audio(y, rate=sr))
                    stft = np.abs(librosa.stft(y))

                    energy = librosa.feature.rms(S=stft)

                    window_len = 51
                    window = np.ones(window_len) / window_len
                    energy_smooth = signal.convolve(
                        energy.squeeze(), window, mode="same"
                    )

                    energy_threshold = np.percentile(energy_smooth, 20)

                    nonsilence_idx = np.where(energy_smooth >= energy_threshold)[0]

                    if nonsilence_idx.size == 0:
                        print(
                            f"No non-silent sections found in {audio_file}, skipping."
                        )
                        print(os.path.join(root, file))
                        continue
                    start_sample = nonsilence_idx[0] * 512
                    end_sample = nonsilence_idx[-1] * 512 + 512
                  
                    audio_nonsilent = y[start_sample:end_sample]

                    os.makedirs(root.replace(src, dst), exist_ok=True)
                    output_path = os.path.join(root.replace(src, dst), file)
                    sf.write(output_path, audio_nonsilent, sr)
                except Exception as e:
                    print(e)
                    print(os.path.join(root, file))
            else:
                os.makedirs(root.replace(src, dst), exist_ok=True)
                shutil.copy(
                    os.path.join(root, file), os.path.join(root.replace(src, dst), file)
                )


def peak_norm_to_0dbfs(y, sr):

    peak_val = np.max(np.abs(y))

    if peak_val == 1.0:
        return y, 1.0
    y_0dbfs = y / peak_val

    return y_0dbfs, peak_val


def run_0dbfs_norm(src_dir, dst_dir):
    for root, dir, files in os.walk(src_dir):
        os.makedirs(root.replace(src_dir, dst_dir), exist_ok=True)
        for file in tqdm(files):
            if file.endswith(".wav"):
                wav_path = os.path.join(root, file)
                y, sr = librosa.load(wav_path, sr=None)
                y_0dbfs, norm_factor = peak_norm_to_0dbfs(y, sr)
                output_path = os.path.join(root.replace(src_dir, dst_dir), file)
                sf.write(output_path, y_0dbfs, sr)
                shutil.copy(
                    wav_path.replace(".wav", ".txt"),
                    output_path.replace(".wav", ".txt"),
                )


if __name__ == "main":
    """TODO..."""
