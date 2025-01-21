import pandas as pd
import torch
from torch.utils.data.dataloader import Dataset
from RawBoost import (
    ISD_additive_noise,
    LnL_convolutive_noise,
    SSI_additive_noise,
    normWav,
)
import soundfile as sf
import numpy as np


def process_Rawboost_feature(feature, sr, args, algo):

    # Data process by Convolutive noise (1st algo)
    if algo == 1:

        feature = LnL_convolutive_noise(
            feature,
            args.N_f,
            args.nBands,
            args.minF,
            args.maxF,
            args.minBW,
            args.maxBW,
            args.minCoeff,
            args.maxCoeff,
            args.minG,
            args.maxG,
            args.minBiasLinNonLin,
            args.maxBiasLinNonLin,
            sr,
        )

    # Data process by Impulsive noise (2nd algo)
    elif algo == 2:

        feature = ISD_additive_noise(feature, args.P, args.g_sd)

    # Data process by coloured additive noise (3rd algo)
    elif algo == 3:

        feature = SSI_additive_noise(
            feature,
            args.SNRmin,
            args.SNRmax,
            args.nBands,
            args.minF,
            args.maxF,
            args.minBW,
            args.maxBW,
            args.minCoeff,
            args.maxCoeff,
            args.minG,
            args.maxG,
            sr,
        )

    # Data process by all 3 algo. together in series (1+2+3) 设置这个
    elif algo == 4:
        # print("ok")

        feature = LnL_convolutive_noise(
            feature,
            args.N_f,
            args.nBands,
            args.minF,
            args.maxF,
            args.minBW,
            args.maxBW,
            args.minCoeff,
            args.maxCoeff,
            args.minG,
            args.maxG,
            args.minBiasLinNonLin,
            args.maxBiasLinNonLin,
            sr,
        )
        feature = ISD_additive_noise(feature, args.P, args.g_sd)
        feature = SSI_additive_noise(
            feature,
            args.SNRmin,
            args.SNRmax,
            args.nBands,
            args.minF,
            args.maxF,
            args.minBW,
            args.maxBW,
            args.minCoeff,
            args.maxCoeff,
            args.minG,
            args.maxG,
            sr,
        )

    # Data process by 1st two algo. together in series (1+2)
    elif algo == 5:

        feature = LnL_convolutive_noise(
            feature,
            args.N_f,
            args.nBands,
            args.minF,
            args.maxF,
            args.minBW,
            args.maxBW,
            args.minCoeff,
            args.maxCoeff,
            args.minG,
            args.maxG,
            args.minBiasLinNonLin,
            args.maxBiasLinNonLin,
            sr,
        )
        feature = ISD_additive_noise(feature, args.P, args.g_sd)

    # Data process by 1st and 3rd algo. together in series (1+3)
    elif algo == 6:

        feature = LnL_convolutive_noise(
            feature,
            args.N_f,
            args.nBands,
            args.minF,
            args.maxF,
            args.minBW,
            args.maxBW,
            args.minCoeff,
            args.maxCoeff,
            args.minG,
            args.maxG,
            args.minBiasLinNonLin,
            args.maxBiasLinNonLin,
            sr,
        )
        feature = SSI_additive_noise(
            feature,
            args.SNRmin,
            args.SNRmax,
            args.nBands,
            args.minF,
            args.maxF,
            args.minBW,
            args.maxBW,
            args.minCoeff,
            args.maxCoeff,
            args.minG,
            args.maxG,
            sr,
        )

    # Data process by 2nd and 3rd algo. together in series (2+3)
    elif algo == 7:

        feature = ISD_additive_noise(feature, args.P, args.g_sd)
        feature = SSI_additive_noise(
            feature,
            args.SNRmin,
            args.SNRmax,
            args.nBands,
            args.minF,
            args.maxF,
            args.minBW,
            args.maxBW,
            args.minCoeff,
            args.maxCoeff,
            args.minG,
            args.maxG,
            sr,
        )

    # Data process by 1st two algo. together in Parallel (1||2)
    elif algo == 8:

        feature1 = LnL_convolutive_noise(
            feature,
            args.N_f,
            args.nBands,
            args.minF,
            args.maxF,
            args.minBW,
            args.maxBW,
            args.minCoeff,
            args.maxCoeff,
            args.minG,
            args.maxG,
            args.minBiasLinNonLin,
            args.maxBiasLinNonLin,
            sr,
        )
        feature2 = ISD_additive_noise(feature, args.P, args.g_sd)

        feature_para = feature1 + feature2
        feature = normWav(feature_para, 0)  # normalized resultant waveform

    # original data without Rawboost processing
    else:

        feature = feature

    return feature


class PrepASV19Dataset(Dataset):
    def __init__(self, protocol_file_path, data_path, args, data_type="time_frame"):
        self.train_protocol = pd.read_csv(protocol_file_path, sep=" ", header=None)
        self.data_path = data_path
        self.data_type = data_type
        self.args = args

    def __len__(self):
        return self.train_protocol.shape[0]

    # def __getitem__(self, index):
    #     # data_file_path = self.data_path + self.train_protocol.iloc[index, 1]
    #     data_file_path = self.data_path + self.train_protocol.iloc[index, 0]

    #     if self.data_type == "time_frame":
    #         # sample, sr = sf.read(data_file_path + ".flac")
    #         sample, sr = sf.read(data_file_path)
    #         sample = process_Rawboost_feature(sample, sr, self.args, 0)
    #         sample = torch.tensor(sample, dtype=torch.float32)
    #         sample = torch.unsqueeze(sample, 0)
    #         # label = self.train_protocol.iloc[index, 4]
    #         label = self.train_protocol.iloc[index, 1]
    #         label = label_encode(label)
    #         # sub_class = self.train_protocol.iloc[index, 3]
    #         # sub_class = sub_class_encode_19(sub_class)
    #         # return sample, label, sub_class
    #         return sample, label

    #     if self.data_type == "CQT":
    #         sample = torch.load(data_file_path + ".pt")
    #         sample = torch.tensor(sample, dtype=torch.float32)
    #         sample = torch.unsqueeze(sample, 0)
    #         label = self.train_protocol.iloc[index, 4]
    #         label = label_encode(label)
    #         sub_class = self.train_protocol.iloc[index, 3]
    #         sub_class = sub_class_encode_19(sub_class)
    #         return sample, label, sub_class

    def __getitem__(self, index):

        data_file_path = self.data_path + self.train_protocol.iloc[index, 1]

        # print(123)
        if self.data_type == "time_frame":
            try:
                x, sr = sf.read(data_file_path)
                # sample = process_Rawboost_feature(sample, 16000, self.args, 4)
            except Exception as e:
                print(f"error: {e}")
            
            if len(x) < 6 * 16000:
                x = np.tile(x, int((6 * 16000) // len(x)) + 1)
            x = x[0 : (int(6 * 16000))]
            sample=x
            sample = torch.tensor(sample, dtype=torch.float32)
            sample = torch.unsqueeze(sample, 0)
            label = self.train_protocol.iloc[index, 4]
            label = label_encode(label)

            sub_class = self.train_protocol.iloc[index, 1]
            sub_variant = self.train_protocol.iloc[index, 0]
            sub_source = self.train_protocol.iloc[index, 3]
            # sub_class = sub_class_encode_19(sub_class)
            # print(sample)
            # exit()
            return sample, label, sub_class, sub_source, sub_variant

        if self.data_type == "CQT":
            sample = torch.load(data_file_path + ".pt")
            sample = torch.tensor(sample, dtype=torch.float32)
            sample = torch.unsqueeze(sample, 0)
            label = self.train_protocol.iloc[index, 4]
            label = label_encode(label)
            sub_class = self.train_protocol.iloc[index, 1]
            sub_class = sub_class_encode_19(sub_class)
            return sample, label, sub_class, sub_source, sub_variant

    def get_weights(self):
        label_info = self.train_protocol.iloc[:, 1]
        num_zero_class = (label_info == "bonafide").sum()
        num_one_class = (label_info == "spoof").sum()
        weights = torch.tensor([num_one_class, num_zero_class], dtype=torch.float32)
        weights = weights / (weights.sum())
        return weights


class PrepASV15Dataset(Dataset):
    def __init__(self, protocol_file_path, data_path, data_type="time_frame"):
        self.train_protocol = pd.read_csv(protocol_file_path, sep=" ", header=None)
        self.data_path = data_path
        self.data_type = data_type

    def __len__(self):
        return self.train_protocol.shape[0]

    def __getitem__(self, index):
        data_file_path = self.data_path + self.train_protocol.iloc[index, 1]

        if self.data_type == "time_frame":
            sample, _ = sf.read(data_file_path)
            sample = torch.tensor(sample, dtype=torch.float32)
            sample = torch.unsqueeze(sample, 0)
            label = self.train_protocol.iloc[index, 4]
            label = label_encode(label)
            sub_class = self.train_protocol.iloc[index, 2]
            sub_class = sub_class_encode_15(sub_class)
            sub_source = self.train_protocol.iloc[index, 3]
            sub_variant = self.train_protocol.iloc[index, 0]
            return sample, label, sub_class, sub_source, sub_variant

        if self.data_type == "CQT":
            sample = torch.load(data_file_path + ".pt")
            sample = torch.tensor(sample, dtype=torch.float32)
            sample = torch.unsqueeze(sample, 0)
            label = self.train_protocol.iloc[index, 4]
            label = label_encode(label)
            sub_class = self.train_protocol.iloc[index, 2]
            sub_class = sub_class_encode_15(sub_class)
            sub_variant = self.train_protocol.iloc[index, 0]
            sub_source = self.train_protocol.iloc[index, 3]
            return sample, label, sub_class, sub_source, sub_variant

    def get_weights(self):
        label_info = self.train_protocol.iloc[:, 3]
        num_zero_class = (label_info == "human").sum()
        num_one_class = (label_info == "spoof").sum()
        weights = torch.tensor([num_one_class, num_zero_class], dtype=torch.float32)
        weights = weights / (weights.sum())
        return weights


def label_encode(label):
    if label == "bonafide":
        label = torch.tensor(0, dtype=torch.int64)
    elif label == "human":
        label = torch.tensor(0, dtype=torch.int64)
    else:
        label = torch.tensor(1, dtype=torch.int64)
    return label


def sub_class_encode_19(label):
    if label == "-":
        label = torch.tensor(0, dtype=torch.int64)
    elif label == "A01":
        label = torch.tensor(1, dtype=torch.int64)
    elif label == "A02":
        label = torch.tensor(2, dtype=torch.int64)
    elif label == "A03":
        label = torch.tensor(3, dtype=torch.int64)
    elif label == "A04":
        label = torch.tensor(4, dtype=torch.int64)
    elif label == "A05":
        label = torch.tensor(5, dtype=torch.int64)
    elif label == "A06":
        label = torch.tensor(6, dtype=torch.int64)
    elif label == "A07":
        label = torch.tensor(7, dtype=torch.int64)
    elif label == "A08":
        label = torch.tensor(8, dtype=torch.int64)
    elif label == "A09":
        label = torch.tensor(9, dtype=torch.int64)
    elif label == "A10":
        label = torch.tensor(10, dtype=torch.int64)
    elif label == "A11":
        label = torch.tensor(11, dtype=torch.int64)
    elif label == "A12":
        label = torch.tensor(12, dtype=torch.int64)
    elif label == "A13":
        label = torch.tensor(13, dtype=torch.int64)
    elif label == "A14":
        label = torch.tensor(14, dtype=torch.int64)
    elif label == "A15":
        label = torch.tensor(15, dtype=torch.int64)
    elif label == "A16":
        label = torch.tensor(16, dtype=torch.int64)
    elif label == "A17":
        label = torch.tensor(17, dtype=torch.int64)
    elif label == "A18":
        label = torch.tensor(18, dtype=torch.int64)
    elif label == "A19":
        label = torch.tensor(19, dtype=torch.int64)
    return label


def sub_class_encode_15(label):
    if label == "human":
        label = torch.tensor(0, dtype=torch.int64)
    elif label == "S1":
        label = torch.tensor(1, dtype=torch.int64)
    elif label == "S2":
        label = torch.tensor(2, dtype=torch.int64)
    elif label == "S3":
        label = torch.tensor(3, dtype=torch.int64)
    elif label == "S4":
        label = torch.tensor(4, dtype=torch.int64)
    elif label == "S5":
        label = torch.tensor(5, dtype=torch.int64)
    elif label == "S6":
        label = torch.tensor(6, dtype=torch.int64)
    elif label == "S7":
        label = torch.tensor(7, dtype=torch.int64)
    elif label == "S8":
        label = torch.tensor(8, dtype=torch.int64)
    elif label == "S9":
        label = torch.tensor(9, dtype=torch.int64)
    elif label == "S10":
        label = torch.tensor(10, dtype=torch.int64)
    return label
