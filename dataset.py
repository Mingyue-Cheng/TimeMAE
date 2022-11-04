import numpy as np
import torch
import torch.utils.data as Data


class Dataset(Data.Dataset):
    def __init__(self, device, mode, data, wave_len):
        self.device = device
        self.datas, self.label = data
        self.mode = mode
        self.wave_len = wave_len
        self.__padding__()

    def __padding__(self):
        origin_len = self.datas[0].shape[0]
        if origin_len % self.wave_len:
            padding_len = self.wave_len - (origin_len % self.wave_len)
            padding = np.zeros((len(self.datas), padding_len, self.datas[0].shape[1]), dtype=np.float32)
            self.datas = np.concatenate([self.datas, padding], axis=-2)

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, item):
        data = torch.tensor(self.datas[item]).to(self.device)
        label = self.label[item]
        return data, torch.tensor(label).to(self.device)

    def shape(self):
        return self.datas[0].shape
