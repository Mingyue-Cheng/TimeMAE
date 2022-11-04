import numpy as np
import torch
from scipy.io import arff


def padding_varying_length(data):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i, j, :][np.isnan(data[i, j, :])] = 0
    return data


def load_UCR(Path='data/', folder='Cricket'):
    train_path = Path + folder + '/' + folder + '_TRAIN.arff'
    test_path = Path + folder + '/' + folder + '_TEST.arff'
    TRAIN_DATA = []
    TRAIN_LABEL = []
    label_dict = {}
    label_index = 0
    with open(train_path, encoding='UTF-8', errors='ignore') as f:
        data, meta = arff.loadarff(f)
        f.close()
    if type(data[0][0]) == np.ndarray:  # multivariate
        for index in range(data.shape[0]):
            raw_data = data[index][0]
            raw_label = data[index][1]
            if label_dict.__contains__(raw_label):
                TRAIN_LABEL.append(label_dict[raw_label])
            else:
                label_dict[raw_label] = label_index
                TRAIN_LABEL.append(label_index)
                label_index += 1
            raw_data_list = raw_data.tolist()
            # print(raw_data_list)
            TRAIN_DATA.append(np.array(raw_data_list).astype(np.float32).transpose(-1, 0))

        TEST_DATA = []
        TEST_LABEL = []
        with open(test_path, encoding='UTF-8', errors='ignore') as f:
            data, meta = arff.loadarff(f)
            f.close()
        for index in range(data.shape[0]):
            raw_data = data[index][0]
            raw_label = data[index][1]
            TEST_LABEL.append(label_dict[raw_label])
            raw_data_list = raw_data.tolist()
            TEST_DATA.append(np.array(raw_data_list).astype(np.float32).transpose(-1, 0))

        index = np.arange(0, len(TRAIN_DATA))
        np.random.shuffle(index)

        TRAIN_DATA = padding_varying_length(np.array(TRAIN_DATA))
        TEST_DATA = padding_varying_length(np.array(TEST_DATA))

        return [np.array(TRAIN_DATA)[index], np.array(TRAIN_LABEL)[index]], \
               [np.array(TRAIN_DATA)[index], np.array(TRAIN_LABEL)[index]], \
               [np.array(TEST_DATA), np.array(TEST_LABEL)]

    else:  # univariate
        for index in range(data.shape[0]):
            raw_data = np.array(list(data[index]))[:-1]
            raw_label = data[index][-1]
            if label_dict.__contains__(raw_label):
                TRAIN_LABEL.append(label_dict[raw_label])
            else:
                label_dict[raw_label] = label_index
                TRAIN_LABEL.append(label_index)
                label_index += 1
            TRAIN_DATA.append(np.array(raw_data).astype(np.float32).reshape(-1, 1))

        TEST_DATA = []
        TEST_LABEL = []
        with open(test_path, encoding='UTF-8', errors='ignore') as f:
            data, meta = arff.loadarff(f)
            f.close()
        for index in range(data.shape[0]):
            raw_data = np.array(list(data[index]))[:-1]
            raw_label = data[index][-1]
            TEST_LABEL.append(label_dict[raw_label])
            TEST_DATA.append(np.array(raw_data).astype(np.float32).reshape(-1, 1))

        TRAIN_DATA = padding_varying_length(np.array(TRAIN_DATA))
        TEST_DATA = padding_varying_length(np.array(TEST_DATA))

        return [np.array(TRAIN_DATA), np.array(TRAIN_LABEL)], [np.array(TRAIN_DATA), np.array(TRAIN_LABEL)], [
            np.array(TEST_DATA), np.array(TEST_LABEL)]


def load_HAR(Path='data/HAR/'):
    train = torch.load(Path + 'train.pt')
    val = torch.load(Path + 'val.pt')
    test = torch.load(Path + 'test.pt')
    TRAIN_DATA = train['samples'].transpose(1, 2).float()
    TRAIN_LABEL = train['labels'].long()
    VAL_DATA = val['samples'].transpose(1, 2).float()
    VAL_LABEL = val['labels'].long()
    TEST_DATA = test['samples'].transpose(1, 2).float()
    TEST_LABEL = test['labels'].long()

    ALL_TRAIN_DATA = torch.cat([TRAIN_DATA, VAL_DATA])
    ALL_TRAIN_LABEL = torch.cat([TRAIN_LABEL, VAL_LABEL])
    print('data loaded')

    return [np.array(ALL_TRAIN_DATA), np.array(ALL_TRAIN_LABEL)], [np.array(TRAIN_DATA), np.array(TRAIN_LABEL)], [
        np.array(TEST_DATA), np.array(TEST_LABEL)]


def load_mat(Path='data/AUSLAN/'):
    if 'UWave' in Path:
        train = torch.load(Path + 'train_new.pt')
        test = torch.load(Path + 'test_new.pt')
    else:
        train = torch.load(Path + 'train.pt')
        test = torch.load(Path + 'test.pt')
    TRAIN_DATA = train['samples'].float()
    TRAIN_LABEL = (train['labels'] - 1).long()
    TEST_DATA = test['samples'].float()
    TEST_LABEL = (test['labels'] - 1).long()
    print('data loaded')

    return [np.array(TRAIN_DATA), np.array(TRAIN_LABEL)], [np.array(TRAIN_DATA), np.array(TRAIN_LABEL)], [
        np.array(TEST_DATA), np.array(TEST_LABEL)]
