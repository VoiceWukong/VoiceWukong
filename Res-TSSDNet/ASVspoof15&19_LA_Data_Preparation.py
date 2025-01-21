import soundfile as sf
import pandas as pd
import numpy as np
import torch
import librosa
from scipy import signal
import os

"""
This program generates 
    1) equal-duration time domain raw waveforms
    2) 2D log power of constant Q transform 
from ASVspoof2019 and ASVspoof2015 official datasets, respectively. 
This program is supposed to be run independently for data preparation.

Official dataset download link: https://www.asvspoof.org/database

The CQT parameter settings follow the ones used in:
X. Li et al., "Replay and synthetic speech detection with res2net architecture," in Proc. ICASSP 2021.

"""


# def gen_time_frame_19(
#     protocol_path, read_audio_path, write_audio_path, duration, status: str
# ):
#     sub_path = write_audio_path + status + "_" + str(duration) + "/"
#     if not os.path.exists(sub_path):
#         os.makedirs(sub_path)

#     protocol = pd.read_csv(protocol_path, sep=" ", header=None).values
#     file_index = protocol[:, 1]
#     variant = protocol[:, 0]
#     num_files = protocol.shape[0]
#     total_sample_count = 0

#     for i in range(num_files):
#         x, fs = sf.read(read_audio_path + file_index[i])
#         # print(123)
#         if len(x) < duration * 16000:
#             x = np.tile(x, int((duration * 16000) // len(x)) + 1)
#         x = x[0 : (int(duration * 16000))]
#         total_sample_count += 1
#         # print(os.path.join(sub_path, variant[i]))
#         if not os.path.exists(os.path.join(sub_path, variant[i], "zh")):
#             os.makedirs(os.path.join(sub_path, variant[i], "zh"))
#             # print(os.path.join(sub_path, variant[i])
#         sf.write(sub_path + file_index[i], x, fs)
#     print(
#         "{} pieces {}-second {} samples generated.".format(
#             total_sample_count, duration, status
#         )
#     )


def gen_time_frame_19(
    protocol_path, read_audio_path, write_audio_path, duration, status: str
):
    sub_path = write_audio_path + status + "_" + str(duration) + "/"
    if not os.path.exists(sub_path):
        os.makedirs(sub_path)

    protocol = pd.read_csv(protocol_path, sep=" ", header=None).values
    file_index = protocol[:, 0]
    num_files = protocol.shape[0]
    total_sample_count = 0

    for i in range(num_files):
        x, fs = sf.read(read_audio_path + file_index[i])
        if len(x) < duration * fs:
            x = np.tile(x, int((duration * fs) // len(x)) + 1)
        x = x[0 : (int(duration * fs))]
        total_sample_count += 1
        os.makedirs(os.path.dirname(sub_path + file_index[i]), exist_ok=True)
        sf.write(sub_path + file_index[i], x, fs)
    print(
        "{} pieces {}-second {} samples generated.".format(
            total_sample_count, duration, status
        )
    )


def gen_cqt_19(
    protocol_path, read_audio_path, write_audio_path, duration=6.4, status="train"
):
    sub_path = write_audio_path + status + "_" + str(duration) + "_cqt" + "/"
    if not os.path.exists(sub_path):
        os.makedirs(sub_path)

    protocol = pd.read_csv(protocol_path, sep=" ", header=None).values
    file_index = protocol[:, 1]
    num_files = protocol.shape[0]
    total_sample_count = 0
    fs = 16000

    for i in range(num_files):
        x, fs = sf.read(read_audio_path + file_index[i] + ".flac")
        len_sample = int(duration * fs)

        if len(x) < len_sample:
            x = np.tile(x, int(len_sample // len(x)) + 1)

        x = x[0 : int(len_sample - 256)]

        x = signal.lfilter([1, -0.97], [1], x)
        x_cqt = librosa.cqt(
            x,
            sr=fs,
            hop_length=256,
            n_bins=432,
            bins_per_octave=48,
            window="hann",
            fmin=15,
        )
        pow_cqt = np.square(np.abs(x_cqt))
        log_pow_cqt = 10 * np.log10(pow_cqt + 1e-30)
        total_sample_count += 1
        torch.save(log_pow_cqt, sub_path + file_index[i] + ".pt")
    print(
        "{} {} CQT features of {}*{} generated.".format(
            total_sample_count, status, 432, int((duration * fs) // 256)
        )
    )


def gen_time_frame_15(
    protocol_path, read_audio_path, write_audio_path, duration, status: str
):
    sub_path = write_audio_path + status + "_" + str(duration) + "/"
    if not os.path.exists(sub_path):
        os.makedirs(sub_path)

    protocol = pd.read_csv(protocol_path, sep=" ", header=None).values
    file_sub_dir = protocol[:, 0]
    file_index = protocol[:, 1]
    num_files = protocol.shape[0]
    total_sample_count = 0

    for i in range(num_files):
        x, fs = sf.read(
            read_audio_path + file_sub_dir[i] + "/" + file_index[i] + ".wav"
        )
        if len(x) < duration * fs:
            x = np.tile(x, int((duration * fs) // len(x)) + 1)
        x = x[0 : (int(duration * fs))]
        total_sample_count += 1
        sf.write(sub_path + file_index[i] + ".wav", x, fs)
    print(
        "{} pieces {}-second {} samples generated.".format(
            total_sample_count, duration, status
        )
    )


def gen_cqt_15(
    protocol_path, read_audio_path, write_audio_path, duration=6.4, status="train"
):
    sub_path = write_audio_path + status + "_" + str(duration) + "_cqt" + "/"
    if not os.path.exists(sub_path):
        os.makedirs(sub_path)

    protocol = pd.read_csv(protocol_path, sep=" ", header=None).values
    file_index = protocol[:, 1]
    num_files = protocol.shape[0]
    total_sample_count = 0
    fs = 16000

    for i in range(num_files):
        x, fs = sf.read(read_audio_path + file_index[i] + ".wav")
        len_sample = int(duration * fs)

        if len(x) < len_sample:
            x = np.tile(x, int(len_sample // len(x)) + 1)

        x = x[0 : int(len_sample - 256)]

        x = signal.lfilter([1, -0.97], [1], x)
        x_cqt = librosa.cqt(
            x,
            sr=fs,
            hop_length=256,
            n_bins=432,
            bins_per_octave=48,
            window="hann",
            fmin=15,
        )
        pow_cqt = np.square(np.abs(x_cqt))
        log_pow_cqt = 10 * np.log10(pow_cqt + 1e-30)
        total_sample_count += 1
        torch.save(log_pow_cqt, sub_path + file_index[i] + ".pt")
    print(
        "{} {} CQT features of {}*{} generated.".format(
            total_sample_count, status, 432, int((duration * fs) // 256)
        )
    )


if __name__ == "__main__":
    # TODO: ASVspoof2019 data preparation
    # /home/ydoit/AIGC/Eva/LA
    # directory info of ASVspoof2019 dataset
    root_path = "/home/ydoit/AIGC/Eva/CD_ADD/"
    train_protocol_path = root_path + "dev_train.txt"
    dev_protocol_path = root_path + "dev_eva.txt"
    # eval_protocol_path  = root_path + 'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt'
    # eval_protocol_path = (
    #     root_path + "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"
    # )
    train_data_path = root_path  # + "ASVspoof2019_LA_train/flac/"
    dev_data_path = root_path  # + "ASVspoof2019_LA_dev/flac/"
    eval_data_path = root_path  # + #"ASVspoof2019_LA_eval/flac/"

    # create folders for new types of data
    new_data_path = "/home/ydoit/AIGC/Eva/tssdnetCD_ADD/"
    if not os.path.exists(new_data_path):
        os.makedirs(new_data_path)

    # generate equal-duration time-frames examples
    print("Generating time-frame data...")
    time_dur = 6  # in seconds
    gen_time_frame_19(
        train_protocol_path,
        train_data_path,
        new_data_path,
        duration=time_dur,
        status="train",
    )
    gen_time_frame_19(
        dev_protocol_path, dev_data_path, new_data_path, duration=time_dur, status="dev"
    )
    # gen_time_frame_19(
    #     eval_protocol_path,
    #     eval_data_path,
    #     new_data_path,
    #     duration=time_dur,
    #     status="eval",
    # )

    # generate cqt feature per sample
    # print('Generating CQT data...')
    # cqt_dur = 6.4  # in seconds, default ICASSP 2021 setting
    # gen_cqt_19(train_protocol_path, train_data_path, new_data_path, duration=cqt_dur, status='train')
    # gen_cqt_19(dev_protocol_path,   dev_data_path,   new_data_path, duration=cqt_dur, status='dev')
    # gen_cqt_19(eval_protocol_path,  eval_data_path,  new_data_path, duration=cqt_dur, status='eval')

    # TODO: ASVspoof2015 data preparation
    # directory info of ASVspoof2015 dataset
    # root_path = 'F:/ASVspoof2015/'
    # train_protocol_path = root_path + 'CM_protocol/cm_train.trn.txt'
    # dev_protocol_path   = root_path + 'CM_protocol/cm_develop.ndx.txt'
    # eval_protocol_path  = root_path + 'CM_protocol/cm_evaluation.ndx.txt'
    # train_data_path     = root_path + 'wav/'
    # dev_data_path       = root_path + 'wav/'
    # eval_data_path      = root_path + 'wav/'

    # new_data_path = root_path + 'data/'
    # if not os.path.exists(new_data_path):
    #     os.makedirs(new_data_path)

    # time_dur = 6
    # gen_time_frame_15(train_protocol_path, train_data_path, new_data_path, duration=time_dur, status='train')
    # gen_time_frame_15(dev_protocol_path,   dev_data_path,   new_data_path, duration=time_dur, status='dev')
    # gen_time_frame_15(eval_protocol_path,  eval_data_path,  new_data_path, duration=time_dur, status='eval')

    # # generate cqt feature per sample
    # print('Generating CQT data...')
    # cqt_dur = 6.4  # in seconds, default ICASSP 2021 setting
    # gen_cqt_15(train_protocol_path, train_data_path, new_data_path, duration=cqt_dur, status='train')
    # gen_cqt_15(dev_protocol_path,   dev_data_path,   new_data_path, duration=cqt_dur, status='dev')
    # gen_cqt_15(eval_protocol_path,  eval_data_path,  new_data_path, duration=cqt_dur, status='eval')

    print("End of Program.")
