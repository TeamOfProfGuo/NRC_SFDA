import pdb
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


class Selfie(object):
    def __init__(self, args, size_of_data, num_of_classes, history_length, threshold):
        self.args = args
        self.size_of_data = size_of_data
        self.num_of_classes = num_of_classes
        self.history_length = history_length
        self.threshold = threshold

        # initialize the prediction history recorder
        self.histroy_predictions = torch.ones((size_of_data, history_length), dtype=int) * -1

        # Max predictive uncertainty
        if history_length > num_of_classes:
            self.max_certainty = -torch.log(torch.tensor(1.0)/float(num_of_classes))
        else:
            self.max_certainty = -torch.log(torch.tensor(1.0)/float(history_length))

        # Corrected label map
        self.corrected_labels = torch.ones(size_of_data, dtype=int)*-1
        # self.corrected_labels = {}
        # for i in range(size_of_data):
        #     self.corrected_labels[i] = -1

        # self.update_counters = torch.zeros(size_of_data, dtype=int)


    def update_prediction(self, epoch, prediction, start_idx, end_idx):
        column = epoch % self.history_length
        self.histroy_predictions[start_idx:end_idx, column] = prediction


    def separate_clean_and_unclean_samples(self, loss_array, noise_rate):
        # select clean samples base on loss value
        _, indeices = torch.sort(loss_array)
        clean_num = int(np.ceil(float(loss_array.shape[0]) * (1.0 - noise_rate)))  # noise ratio needs to be modified
        # loss = loss_array[:clean_num]
        return indeices[:clean_num]


    # def async_update_prediction_matrix(self, ids, softmax_matrix):
    #     for i in range(len(ids)):
    #         id = ids[i]
    #         predicted_label = np.argmax(softmax_matrix[i])
    #         # append the predicted label to the prediction matrix
    #         cur_index = self.update_counters[id] % self.history_length
    #         self.all_predictions[id][cur_index] = predicted_label
    #         self.update_counters[id] += 1


    def get_refurbishable_samples(self, start_idx, end_idx):
        for cls in range(self.num_of_classes):
            cur_count = torch.sum(torch.where(self.histroy_predictions[start_idx:end_idx]==cls, 1, 0), dim=1).unsqueeze(1) 
            if cls == 0:
                counts = cur_count
            else:
                counts = torch.cat((counts, cur_count), dim=1)

        counts = counts / self.num_of_classes

        log_counts = torch.log(counts)
        log_counts[log_counts == -float("Inf")] = 0

        uncertainty = (-(counts * log_counts).sum(1)) / self.max_certainty   # [b]

        refurbish_idx = torch.where(uncertainty <= self.threshold)[0]

        refurbish_label = counts.argmax(1)    # [b]

        self.corrected_labels[start_idx:end_idx][refurbish_idx] = refurbish_label[refurbish_idx]

        return refurbish_label[refurbish_idx], refurbish_idx


    def merge_clean_and_corrected_samples(self, clean_idx, refurbish_idx, start_idx, end_idx, label):
        # return lst: [refurbish_idx, else_clean_idx]
        else_clean_idx = []
        else_clean_idx_label = []

        for idx in clean_idx:
            if idx in refurbish_idx:
                continue

            else_clean_idx.append(idx)

            if self.corrected_labels[start_idx:end_idx][idx] != -1:
                # if the sample was corrected at previous epoch, we reuse the corrected label for current mini-batch
                else_clean_idx_label.append(self.corrected_labels[start_idx:end_idx][idx])
            else:
                else_clean_idx_label.append(label[idx])

        return [torch.tensor(else_clean_idx).cuda(), torch.tensor(else_clean_idx_label).cuda()]


    def patch_clean_with_refurbishable_sample_batch(self, loss_array, noise_rate, start_idx, end_idx, label):
        # 1. separate clean and unclean samples
        clean_idx = self.separate_clean_and_unclean_samples(loss_array, noise_rate).cuda()
        # 2. get refurbishable samples
        refurbish_label, refurbish_idx = self.get_refurbishable_samples(start_idx, end_idx)
        refurbish_label = refurbish_label.cuda()
        refurbish_idx = refurbish_idx.cuda()
        # 3. merging
        return [refurbish_idx, refurbish_label], self.merge_clean_and_corrected_samples(clean_idx, refurbish_idx, start_idx, end_idx, label)


    def predictions_clear(self):
        self.all_predictions.clear()
        for i in range(self.size_of_data):
            self.all_predictions[i] = np.zeros(self.history_length, dtype=int)


    def compute_new_noise_ratio(self):
        num_corrected_sample = 0
        for key, value in self.corrected_labels.items():
            if value != -1:
                num_corrected_sample += 1

        return 1.0 - float(num_corrected_sample) / float(self.size_of_data)