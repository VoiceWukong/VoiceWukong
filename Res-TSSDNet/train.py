import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
from data import PrepASV19Dataset, PrepASV15Dataset
import models
from test import asv_cal_accuracies, cal_roc_eer
import os
import sys
import time
import tqdm

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASVspoof2021 baseline system")
    # Dataset
    parser.add_argument(
        "--database_path",
        type=str,
        default="/home/ydoit/AIGC/Eva/LA/",
        help="Change this to user's full directory address of LA database (ASVspoof2019- for training & development (used as validation), ASVspoof2021 for evaluation scores). We assume that all three ASVspoof 2019 LA train, LA dev and ASVspoof2021 LA eval data folders are in the same database_path directory.",
    )
    """
    % database_path/
    %   |- LA
    %      |- ASVspoof2021_LA_eval/flac
    %      |- ASVspoof2019_LA_train/flac
    %      |- ASVspoof2019_LA_dev/flac

 
 
    """

    parser.add_argument(
        "--protocols_path",
        type=str,
        default="/home/ydoit/AIGC/Eva/LA/",
        help="Change with path to user's LA database protocols directory address",
    )
    """
    % protocols_path/
    %   |- ASVspoof_LA_cm_protocols
    %      |- ASVspoof2021.LA.cm.eval.trl.txt
    %      |- ASVspoof2019.LA.cm.dev.trl.txt 
    %      |- ASVspoof2019.LA.cm.train.trn.txt
  
    """

    # Hyperparameters
    parser.add_argument("--batch_size", type=int, default=14)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.000001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--loss", type=str, default="weighted_CCE")
    # model
    parser.add_argument(
        "--seed", type=int, default=1234, help="random seed (default: 1234)"
    )

    parser.add_argument("--model_path", type=str, default=None, help="Model checkpoint")
    parser.add_argument(
        "--comment", type=str, default=None, help="Comment to describe the saved model"
    )
    # Auxiliary arguments
    parser.add_argument(
        "--track", type=str, default="LA", choices=["LA", "PA", "DF"], help="LA/PA/DF"
    )
    parser.add_argument(
        "--eval_output",
        type=str,
        default=None,
        help="Path to save the evaluation result",
    )
    parser.add_argument("--eval", action="store_true", default=False, help="eval mode")
    parser.add_argument(
        "--is_eval", action="store_true", default=False, help="eval database"
    )
    parser.add_argument("--eval_part", type=int, default=0)
    # backend options
    parser.add_argument(
        "--cudnn-deterministic-toggle",
        action="store_false",
        default=True,
        help="use cudnn-deterministic? (default true)",
    )

    parser.add_argument(
        "--cudnn-benchmark-toggle",
        action="store_true",
        default=False,
        help="use cudnn-benchmark? (default false)",
    )

    ##===================================================Rawboost data augmentation ======================================================================#

    parser.add_argument(
        "--algo",
        type=int,
        default=0,  # 5
        help="Rawboost algos discriptions. 0: No augmentation 1: LnL_convolutive_noise, 2: ISD_additive_noise, 3: SSI_additive_noise, 4: series algo (1+2+3), \
                          5: series algo (1+2), 6: series algo (1+3), 7: series algo(2+3), 8: parallel algo(1,2) .[default=0]",
    )

    # LnL_convolutive_noise parameters
    parser.add_argument(
        "--nBands",
        type=int,
        default=5,
        help="number of notch filters.The higher the number of bands, the more aggresive the distortions is.[default=5]",
    )
    parser.add_argument(
        "--minF",
        type=int,
        default=20,
        help="minimum centre frequency [Hz] of notch filter.[default=20] ",
    )
    parser.add_argument(
        "--maxF",
        type=int,
        default=8000,
        help="maximum centre frequency [Hz] (<sr/2)  of notch filter.[default=8000]",
    )
    parser.add_argument(
        "--minBW",
        type=int,
        default=100,
        help="minimum width [Hz] of filter.[default=100] ",
    )
    parser.add_argument(
        "--maxBW",
        type=int,
        default=1000,
        help="maximum width [Hz] of filter.[default=1000] ",
    )
    parser.add_argument(
        "--minCoeff",
        type=int,
        default=10,
        help="minimum filter coefficients. More the filter coefficients more ideal the filter slope.[default=10]",
    )
    parser.add_argument(
        "--maxCoeff",
        type=int,
        default=100,
        help="maximum filter coefficients. More the filter coefficients more ideal the filter slope.[default=100]",
    )
    parser.add_argument(
        "--minG",
        type=int,
        default=0,
        help="minimum gain factor of linear component.[default=0]",
    )
    parser.add_argument(
        "--maxG",
        type=int,
        default=0,
        help="maximum gain factor of linear component.[default=0]",
    )
    parser.add_argument(
        "--minBiasLinNonLin",
        type=int,
        default=5,
        help=" minimum gain difference between linear and non-linear components.[default=5]",
    )
    parser.add_argument(
        "--maxBiasLinNonLin",
        type=int,
        default=20,
        help=" maximum gain difference between linear and non-linear components.[default=20]",
    )
    parser.add_argument(
        "--N_f",
        type=int,
        default=5,
        help="order of the (non-)linearity where N_f=1 refers only to linear components.[default=5]",
    )

    # ISD_additive_noise parameters
    parser.add_argument(
        "--P",
        type=int,
        default=10,
        help="Maximum number of uniformly distributed samples in [%].[defaul=10]",
    )
    parser.add_argument(
        "--g_sd", type=int, default=2, help="gain parameters > 0. [default=2]"
    )

    # SSI_additive_noise parameters
    parser.add_argument(
        "--SNRmin",
        type=int,
        default=10,
        help="Minimum SNR value for coloured additive noise.[defaul=10]",
    )
    parser.add_argument(
        "--SNRmax",
        type=int,
        default=40,
        help="Maximum SNR value for coloured additive noise.[defaul=40]",
    )
    args = parser.parse_args()
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # TODO: Define dataset scope and data type
    # specify the data type and root path
    dataset = 19  # {'ASVspoof2019': 19, 'ASVspoof2015': 15}
    data_type = "time_frame"  # {'time_frame', 'CQT'}

    if not os.path.exists("./trained_models_muiltdomain/"):
        os.makedirs("./trained_models_muiltdomain/")

    if data_type == "time_frame":
        if dataset == 15:
            root_path = "F:/ASVspoof2015/"
            train_protocol_file_path = root_path + "CM_protocol/cm_train.trn.txt"
            dev_protocol_file_path = root_path + "CM_protocol/cm_develop.ndx.txt"
            eval_protocol_file_path = root_path + "CM_protocol/cm_evaluation.ndx.txt"
            train_data_path = root_path + "data/train_6/"
            dev_data_path = root_path + "data/dev_6/"
            eval_data_path = root_path + "data/eval_6/"
        else:
            root_path = "/home/ydoit/AIGC/Eva/tssdnetCD_ADD/"

            train_protocol_file_path = "/home/ydoit/AIGC/Eva/CD_ADD/dev_train.txt"
            dev_protocol_file_path = "/home/ydoit/AIGC/Eva/CD_ADD/dev_eva.txt"
            eval_protocol_file_path = "/home/ydoit/AIGC/Eva/CD_ADD/dev_eva.txt"
            train_data_path = "/home/ydoit/AIGC/Eva/tssdnetCD_ADD/train_6/"
            dev_data_path = "/home/ydoit/AIGC/Eva/tssdnetCD_ADD/dev_6/"
            eval_data_path = "/home/ydoit/AIGC/Eva/tssdnetCD_ADD/dev_6/"

    elif data_type == "CQT":
        if dataset == 15:
            root_path = "F:/ASVspoof2015/"
            train_protocol_file_path = root_path + "CM_protocol/cm_train.trn.txt"
            dev_protocol_file_path = root_path + "CM_protocol/cm_develop.ndx.txt"
            eval_protocol_file_path = root_path + "CM_protocol/cm_evaluation.ndx.txt"
            train_data_path = root_path + "data/train_6.4_cqt/"
            dev_data_path = root_path + "data/dev_6.4_cqt/"
            eval_data_path = root_path + "data/eval_6.4_cqt/"
        else:
            root_path = "F:/ASVspoof2019/LA/"
            train_protocol_file_path = (
                root_path
                + "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"
            )
            dev_protocol_file_path = (
                root_path
                + "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt"
            )
            eval_protocol_file_path = (
                root_path
                + "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"
            )
            train_data_path = root_path + "data/train_6.4_cqt/"
            dev_data_path = root_path + "data/dev_6.4_cqt/"
            eval_data_path = root_path + "data/eval_6.4_cqt/"
    else:
        print("Program only supports 'time_frame' and 'CQT' data types.")
        sys.exit()

    # TODO: Prepare data and set training parameters
    if dataset == 15:
        train_set = PrepASV15Dataset(
            train_protocol_file_path, train_data_path, data_type=data_type
        )
    else:
        train_set = PrepASV19Dataset(
            train_protocol_file_path, train_data_path, args=args, data_type=data_type
        )
    weights = train_set.get_weights().to(device)  # weight used for WCE
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)

    if data_type == "CQT":
        Net = models.SSDNet2D()  # 2D-Res-TSSDNet
    else:
        Net = models.SSDNet1D()  # Res-TSSDNet
        # Net = models.DilatedNet()  # Inc-TSSDNet
    Net = Net.to(device)

    num_total_learnable_params = sum(
        i.numel() for i in Net.parameters() if i.requires_grad
    )
    print("Number of learnable params: {}.".format(num_total_learnable_params))

    optimizer = optim.Adam(Net.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    loss_type = "WCE"  # {'WCE', 'mixup'}
    check_point = torch.load(
        "/home/ydoit/AIGC/Eva/tssdnet/end-to-end-synthetic-speech-detection/pretrained/Res_TSSDNet_time_frame_61_ASVspoof2019_LA_Loss_0.0017_dEER_0.74%_eEER_1.64%.pth"
    )
    Net.load_state_dict(check_point["model_state_dict"])
    print(
        "Model loaded : {}".format(
            "/home/ydoit/AIGC/Eva/tssdnet/end-to-end-synthetic-speech-detection/pretrained/Res_TSSDNet_time_frame_61_ASVspoof2019_LA_Loss_0.0017_dEER_0.74%_eEER_1.64%.pth"
        )
    )
    # TODO: Training
    print(
        "Training data: {}, Date type: {}. Training started...".format(
            train_data_path, data_type
        )
    )

    num_epoch = 100
    loss_per_epoch = torch.zeros(
        num_epoch,
    )
    best_d_eer = [0.09, 0]

    if not os.path.exists("./trained_models_muiltdomain/train_log/"):
        os.makedirs("./trained_models_muiltdomain/train_log")

    log_path = "./trained_models_muiltdomain/train_log"
    time_name = time.ctime()
    time_name = time_name.replace(" ", "_")
    time_name = time_name.replace(":", "_")
    f = open(log_path + time_name + ".csv", "w+")

    # for epoch in range(check_point['epoch']+1, num_epoch):
    for epoch in tqdm.tqdm(range(num_epoch)):
        Net.train()
        t = time.time()
        total_loss = 0
        counter = 0
        for batch in train_loader:
            counter += 1
            # forward
            # samples, labels, _ = batch
            samples, labels = batch
            samples = samples.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            if loss_type == "mixup":
                # mixup
                alpha = 0.1
                lam = np.random.beta(alpha, alpha)
                lam = torch.tensor(lam, requires_grad=False)
                index = torch.randperm(len(labels))
                samples = lam * samples + (1 - lam) * samples[index, :]
                preds = Net(samples)
                labels_b = labels[index]
                loss = lam * F.cross_entropy(preds, labels) + (
                    1 - lam
                ) * F.cross_entropy(preds, labels_b)
            else:
                preds = Net(samples)
                loss = F.cross_entropy(preds, labels, weight=weights)
                print(loss)
                exit()
                # loss = F.cross_entropy(preds, labels)

            # backward
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        loss_per_epoch[epoch] = total_loss / counter

        dev_accuracy, d_probs = asv_cal_accuracies(
            dev_protocol_file_path,
            dev_data_path,
            Net,
            device,
            args=args,
            data_type=data_type,
            dataset=dataset,
        )
        d_eer = cal_roc_eer(d_probs, show_plot=False)
        if d_eer <= best_d_eer[0]:
            best_d_eer[0] = d_eer
            best_d_eer[1] = int(epoch)

            eval_accuracy, e_probs = asv_cal_accuracies(
                eval_protocol_file_path,
                eval_data_path,
                Net,
                device,
                args=args,
                data_type=data_type,
                dataset=dataset,
            )
            e_eer = cal_roc_eer(e_probs, show_plot=False)
        else:
            e_eer = 0.99
            eval_accuracy = 0.00

        net_str = (
            data_type
            + "_"
            + str(epoch)
            + "_"
            + "CD_ADD"
            + "_LA_Loss_"
            + str(round(total_loss / counter, 4))
            + "_dEER_"
            + str(round(d_eer * 100, 2))
            + "%_eEER_"
            + str(round(e_eer * 100, 2))
            + "%.pth"
        )
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": Net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": loss_per_epoch,
            },
            ("./trained_models_muiltdomain/" + net_str),
        )

        elapsed = time.time() - t

        print_str = (
            "Epoch: {}, Elapsed: {:.2f} mins, lr: {:.3f}e-3, Loss: {:.4f}, d_acc: {:.2f}%, e_acc: {:.2f}%, "
            "dEER: {:.2f}%, eEER: {:.2f}%, best_dEER: {:.2f}% from epoch {}.".format(
                epoch,
                elapsed / 60,
                optimizer.param_groups[0]["lr"] * 1000,
                total_loss / counter,
                dev_accuracy * 100,
                eval_accuracy * 100,
                d_eer * 100,
                e_eer * 100,
                best_d_eer[0] * 100,
                int(best_d_eer[1]),
            )
        )
        print(print_str)
        df = pd.DataFrame([print_str])
        df.to_csv(
            log_path + time_name + ".csv", sep=" ", mode="a", header=False, index=False
        )

        scheduler.step()

    f.close()
    # plt.plot(torch.log10(loss_per_epoch))
    # plt.show()

    print("End of Program.")
