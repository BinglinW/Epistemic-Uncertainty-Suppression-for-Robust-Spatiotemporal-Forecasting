import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def search_recent_data(train, label_start_idx, T_p, T_h):
    """
    Return (history_range, label_range)
    history_range: (start_idx, end_idx) for history [T_h]
    label_range:   (label_start_idx, label_start_idx + T_p)
    """
    if label_start_idx + T_p > len(train):
        return None
    start_idx = label_start_idx - T_h
    end_idx = label_start_idx
    if start_idx < 0 or end_idx < 0:
        return None
    return (start_idx, end_idx), (label_start_idx, label_start_idx + T_p)


class BoeingDataset(Dataset):
    def __init__(self, clean_data, data_range, config):
        self.T_h = config.model.T_h
        self.T_p = config.model.T_p
        self.V = config.model.V
        self.points_per_hour = config.data.points_per_hour
        self.data_range = data_range
        self.data_name = ""

        self.label = np.array(clean_data)    # (T_total, V, D)
        self.feature = np.array(clean_data)  # (T_total, V, D)

        self.idx_lst = self.get_idx_lst()
        print("sample num:", len(self.idx_lst))

    def __len__(self):
        return len(self.idx_lst)

    def get_time_pos(self, idx):
        idx = np.array(range(self.T_h)) + idx
        pos_w = (idx // (self.points_per_hour * 24)) % 7
        pos_d = idx % (self.points_per_hour * 24)
        return pos_w, pos_d

    def get_idx_lst(self):
        idx_lst = []
        start = self.data_range[0]
        end = self.data_range[1] if self.data_range[1] != -1 else self.feature.shape[0]

        for label_start_idx in range(start, end):
            recent = search_recent_data(self.feature, label_start_idx, self.T_p, self.T_h)
            if recent:
                idx_lst.append(recent)
        return idx_lst

    def __getitem__(self, index):
        recent_idx = self.idx_lst[index]

        # label (future)
        ls, le = recent_idx[1][0], recent_idx[1][1]
        label = self.label[ls:le]

        # history
        hs, he = recent_idx[0][0], recent_idx[0][1]
        node_feature = self.feature[hs:he]

        pos_w, pos_d = self.get_time_pos(hs)
        pos_w = np.array(pos_w, dtype=np.int32)
        pos_d = np.array(pos_d, dtype=np.int32)

        # edge placeholder (not used in demo)
        edge = torch.Tensor([1, 1, 1] * len(label))

        return label, node_feature, pos_w, pos_d, edge


def key2value(data, key2value_map):
    return [float(key2value_map[x]) for x in data]


class DataPreprocess:
    def __init__(self, file_path, max_node_dim, node_index_list):
        self.data = self.read_data(file_path, max_node_dim, node_index_list)
        self.data = self.normalization(self.data).astype("float32")

    def get_data(self):
        return self.data

    @staticmethod
    def read_data(file_path, max_node_dim, node_index_list):
        data = pd.read_csv(file_path)
        max_data_len = len(data)
        final_data = []
        mask_value = 0.0

        str_key = ["Install.Type", "Borough", "ntacode"]

        map_1 = {k: i for i, k in enumerate(set(data[str_key[0]]))}
        map_2 = {k: i for i, k in enumerate(set(data[str_key[1]]))}
        map_3 = {k: i for i, k in enumerate(set(data[str_key[2]]))}

        for node_index in node_index_list:
            tem_data = []
            for i in range(max_node_dim):
                if i < len(node_index):
                    if node_index[i] == "Day":
                        timestamp = data[node_index[i]].values.tolist()
                        time_feature1, time_feature2, time_feature3 = [], [], []
                        for time_str in timestamp:
                            # e.g., "9/28/2024"
                            time_values = str(time_str).split("/")
                            time_feature1.append(int(time_values[0]))
                            time_feature2.append(int(time_values[1]))
                            time_feature3.append(int(time_values[2]))
                        tem_data.append(time_feature1)
                        tem_data.append(time_feature2)
                        tem_data.append(time_feature3)
                        break
                    else:
                        if node_index[i] == str_key[0]:
                            tem_data.append(key2value(data[node_index[i]].values.tolist(), map_1))
                        elif node_index[i] == str_key[1]:
                            tem_data.append(key2value(data[node_index[i]].values.tolist(), map_2))
                        elif node_index[i] == str_key[2]:
                            tem_data.append(key2value(data[node_index[i]].values.tolist(), map_3))
                        else:
                            tem_data.append(data[node_index[i]].values.tolist())
                else:
                    tem_data.append([mask_value] * max_data_len)

            final_data.append(tem_data)

        data_array = np.array(final_data)          # (V, D, T)
        data_array = data_array.transpose(2, 0, 1) # (T, V, D)
        return data_array.tolist()

    def normalization(self, feature):
        feature_array = np.array(feature)
        _, node_num, feature_num = feature_array.shape

        self.mean = [[0.0 for _ in range(feature_num)] for _ in range(node_num)]
        self.std = [[0.0 for _ in range(feature_num)] for _ in range(node_num)]

        for i in range(node_num):
            for j in range(feature_num):
                feature_array[:, i, j] = self.normalization_one_feature(feature_array[:, i, j], i, j)

        return feature_array

    def normalization_one_feature(self, feature, node_index, feature_index):
        train = np.nan_to_num(feature)
        mean = np.mean(train)
        std = np.std(train)

        self.mean[node_index][feature_index] = float(mean)
        self.std[node_index][feature_index] = float(std)

        if std == 0:
            return feature
        return (feature - mean) / std

    def reverse_normalization(self, x):
        x = np.array(x)
        _, _, node_num, feature_num = x.shape
        for i in range(node_num):
            for j in range(feature_num):
                x[:, :, i, j] = self.reverse_normalization_one_feature(x[:, :, i, j], i, j)
        return x

    def reverse_normalization_one_feature(self, x, node_index, feature_index):
        return self.mean[node_index][feature_index] + self.std[node_index][feature_index] * x
