import torch
from torch.utils.data.dataloader import DataLoader
from data import PrepASV15Dataset, PrepASV19Dataset
import models
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
from tqdm import tqdm


# def asv_cal_accuracies(
#     protocol,
#     path_data,
#     net,
#     device,
#     args,
#     data_type="time_frame",
#     dataset=19,
#     output_path=None,
# ):
#     net = net.to(device)
#     net.eval()
#     with torch.no_grad():
#         softmax_acc = 0
#         num_files = 0
#         probs = torch.empty(0, 3).to(device)

#         if dataset == 15:
#             test_set = PrepASV15Dataset(protocol, path_data, data_type=data_type)
#         else:
#             test_set = PrepASV19Dataset(protocol, path_data, args, data_type=data_type)

#         test_loader = DataLoader(test_set, batch_size=24, shuffle=False, num_workers=4)

#         zero_list = []
#         one_list = []
#         label_list = []
#         class_list = []
#         source_list = []
#         variant_list = []
#         start_time = time.time()
#         # for test_batch in test_loader:
#         for test_batch in tqdm(test_loader, desc="Processing test batches"):
#             # load batch and infer
#             # print('abc')
#             test_sample, test_label, sub_class, sub_source, sub_variant = test_batch

#             num_files += len(test_label)

#             test_sample = test_sample.to(device)
#             test_label = test_label.to(device)

#             infer = net(test_sample)

#             t1 = F.softmax(infer, dim=1)

#             t2 = test_label.unsqueeze(-1)
#             row = torch.cat((t1, t2), dim=1)
#             """第一列为预测为0的结果  第二列预测为1的结果 第三列真假标签"""
#             probs = torch.cat((probs, row), dim=0)
#             zero_probs = probs[:, 0].tolist()
#             one_probs = probs[:, 1].tolist()
#             label_probs = test_label.tolist()
#             class_probs = [i for i in sub_class]
#             source_probs = [i for i in sub_source]
#             variant_probs = [i for i in sub_variant]
#             zero_list.extend(zero_probs)
#             one_list.extend(one_probs)
#             label_list.extend(label_probs)
#             class_list.extend(class_probs)
#             source_list.extend(source_probs)
#             variant_list.extend(variant_probs)

#             # for i in probs :
#             #     print(i)
#             # break

#             infer = infer.argmax(dim=1)
#             # for i in infer :
#             #     print(i)
#             # break
#             batch_acc = infer.eq(test_label).sum().item()
#             softmax_acc += batch_acc

#         softmax_acc = softmax_acc / num_files
#         end_time = time.time()
#         print("Use time: {:.4f}s.".format(end_time - start_time))
#         with open(output_path, "w") as f:
#             for z, o, l, c, s, v in zip(
#                 zero_list, one_list, label_list, class_list, source_list, variant_list
#             ):
#                 f.write(f"{z} {o} {l} {c} {s} {v}\n")
#     return softmax_acc, probs.to("cpu")


# def cal_roc_eer(probs, show_plot=True):
#     """
#     probs: tensor, number of samples * 3, containing softmax probabilities
#     row wise: [genuine prob, fake prob, label]
#     TP: True Fake
#     FP: False Fake
#     """
#     all_labels = probs[:, 2]
#     """0是真1是假"""
#     """找出所有真假的真假索引"""
#     zero_index = torch.nonzero((all_labels == 0)).squeeze(-1)
#     one_index = torch.nonzero(all_labels).squeeze(-1)
#     """利用索引找到真假概率值"""
#     zero_probs = probs[zero_index, 0]
#     one_probs = probs[one_index, 0]
#     # print(type(zero_probs))
#     # for i in zero_probs:
#     #     print(i)
#     # for i in one_probs:
#     #     print(i)

#     threshold_index = torch.linspace(-0.1, 1.01, 10000)
#     tpr = torch.zeros(
#         len(threshold_index),
#     )
#     fpr = torch.zeros(
#         len(threshold_index),
#     )
#     cnt = 0
#     for i in threshold_index:
#         """第一行是为0的概率，小于等于i会被认为是1假"""
#         tpr[cnt] = one_probs.le(i).sum().item() / len(one_probs)
#         fpr[cnt] = zero_probs.le(i).sum().item() / len(zero_probs)
#         cnt += 1

#     sum_rate = tpr + fpr
#     distance_to_one = torch.abs(sum_rate - 1)
#     eer_index = distance_to_one.argmin(dim=0).item()
#     out_eer = 0.5 * (fpr[eer_index] + 1 - tpr[eer_index]).numpy()

#     if show_plot:
#         print("EER: {:.4f}%.".format(out_eer * 100))
#         plt.figure(1)
#         plt.plot(
#             torch.linspace(-0.2, 1.2, 1000),
#             torch.histc(zero_probs, bins=1000, min=-0.2, max=1.2) / len(zero_probs),
#         )
#         plt.plot(
#             torch.linspace(-0.2, 1.2, 1000),
#             torch.histc(one_probs, bins=1000, min=-0.2, max=1.2) / len(one_probs),
#         )
#         plt.xlabel("Probability of 'Genuine'")
#         plt.ylabel("Per Class Ratio")
#         plt.legend(["Real", "Fake"])
#         plt.grid()

#         plt.figure(3)
#         plt.scatter(fpr, tpr)
#         plt.xlabel("False Positive (Fake) Rate")
#         plt.ylabel("True Positive (Fake) Rate")
#         plt.grid()
#         plt.show()


#     return out_eer
def asv_cal_accuracies(
    protocol, path_data, net, device, args, data_type="time_frame", dataset=19
):
    net = net.to(device)
    net.eval()
    with torch.no_grad():
        softmax_acc = 0
        num_files = 0
        probs = torch.empty(0, 3).to(device)

        if dataset == 15:
            test_set = PrepASV15Dataset(protocol, path_data, data_type=data_type)
        else:
            test_set = PrepASV19Dataset(protocol, path_data, args, data_type=data_type)

        test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=4)

        for test_batch in test_loader:
            # load batch and infer
            # test_sample, test_label, sub_class = test_batch
            test_sample, test_label = test_batch

            # # sub_class level test, comment if unwanted
            # # train & dev 0~6; eval 7~19
            # # selected_index = torch.nonzero(torch.logical_xor(sub_class == 10, sub_class == 0))[:, 0]
            # selected_index = torch.nonzero(sub_class.ne(10))[:, 0]
            # if len(selected_index) == 0:
            #     continue
            # test_sample = test_sample[selected_index, :, :]
            # test_label = test_label[selected_index]

            num_files += len(test_label)
            test_sample = test_sample.to(device)
            test_label = test_label.to(device)
            infer = net(test_sample)

            # obtain output probabilities
            t1 = F.softmax(infer, dim=1)
            t2 = test_label.unsqueeze(-1)
            row = torch.cat((t1, t2), dim=1)
            probs = torch.cat((probs, row), dim=0)

            # calculate example level accuracy
            infer = infer.argmax(dim=1)
            batch_acc = infer.eq(test_label).sum().item()
            softmax_acc += batch_acc

        softmax_acc = softmax_acc / num_files

    return softmax_acc, probs.to("cpu")


def cal_roc_eer(probs, show_plot=False):
    """
    probs: tensor, number of samples * 3, containing softmax probabilities
    row wise: [genuine prob, fake prob, label]
    TP: True Fake
    FP: False Fake
    """
    all_labels = probs[:, 2]
    zero_index = torch.nonzero((all_labels == 0)).squeeze(-1)  # 实际为真的索引
    one_index = torch.nonzero(all_labels).squeeze(-1)  # 实际为假的索引
    zero_probs = probs[zero_index, 0]
    one_probs = probs[one_index, 0]

    threshold_index = torch.linspace(-0.1, 1.01, 10000)
    tpr = torch.zeros(
        len(threshold_index),
    )
    fpr = torch.zeros(
        len(threshold_index),
    )
    cnt = 0
    for i in threshold_index:
        tpr[cnt] = one_probs.le(i).sum().item() / len(
            one_probs
        )  # True Positive Rate  第一列输出是为假的概率 这是统计实际为真  预测为真
        fpr[cnt] = zero_probs.le(i).sum().item() / len(
            zero_probs
        )  # False Positive Rate  第一列输出是为假的概率 这是统计实际为假  预测为真
        cnt += 1

    sum_rate = tpr + fpr
    distance_to_one = torch.abs(sum_rate - 1)
    eer_index = distance_to_one.argmin(dim=0).item()
    out_eer = 0.5 * (fpr[eer_index] + 1 - tpr[eer_index]).numpy()

    if show_plot:
        print("EER: {:.4f}%.".format(out_eer * 100))
        plt.figure(1)
        plt.plot(
            torch.linspace(-0.2, 1.2, 1000),
            torch.histc(zero_probs, bins=1000, min=-0.2, max=1.2) / len(zero_probs),
        )
        plt.plot(
            torch.linspace(-0.2, 1.2, 1000),
            torch.histc(one_probs, bins=1000, min=-0.2, max=1.2) / len(one_probs),
        )
        plt.xlabel("Probability of 'Genuine'")
        plt.ylabel("Per Class Ratio")
        plt.legend(["Real", "Fake"])
        plt.grid()

        plt.figure(3)
        plt.scatter(fpr, tpr)
        plt.xlabel("False Positive (Fake) Rate")
        plt.ylabel("True Positive (Fake) Rate")
        plt.grid()
        plt.show()

    return out_eer


if __name__ == "__main__":

    test_device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    protocol_file_path = "/home/ydoit/AIGC/Eva/tssdnet/end-to-end-synthetic-speech-detection/eval_list.txt"
    data_path = "/home/ydoit/AIGC/VoiceBench/tssdnet/eval_6/"

    # 'F:/ASVSpoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'
    # 'F:/ASVSpoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'
    # 'F:/ASVSpoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt'

    # protocol_file_path = 'F:/ASVspoof2015/CM_protocol/cm_develop.ndx.txt'
    # # cm_train.trn
    # # cm_develop.ndx
    # # cm_evaluation.ndx
    # data_path = 'F:/ASVspoof2015/data/dev_6/'

    Net = models.SSDNet1D()
    num_total_learnable_params = sum(
        i.numel() for i in Net.parameters() if i.requires_grad
    )
    print("Number of learnable params: {}.".format(num_total_learnable_params))

    check_point = torch.load(
        "/home/ydoit/AIGC/Eva/tssdnet/end-to-end-synthetic-speech-detection/trained_models/time_frame_92_ASVspoof2019_LA_Loss_0.0126_dEER_1.89%_eEER_5.04%.pth"
    )
    Net.load_state_dict(check_point["model_state_dict"])

    accuracy, probabilities = asv_cal_accuracies(
        protocol_file_path,
        data_path,
        Net,
        test_device,
        data_type="time_frame",
        dataset=19,
        output_path="/home/ydoit/AIGC/Eva/tssdnet/end-to-end-synthetic-speech-detection/en_eval_score_AD123.txt",
    )
    print(accuracy * 100)

    eer = cal_roc_eer(probabilities)

    print("End of Program.")
