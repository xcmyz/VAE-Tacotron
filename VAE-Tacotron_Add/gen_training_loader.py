import torch
from torch.utils.data import Dataset, DataLoader
from text import text_to_sequence
import hparams

import numpy as np
import os


class SpeechData(Dataset):
    """LJSpeech"""

    def __init__(self, dataset_path):
        self.datasetPath = dataset_path
        self.textPath = os.path.join(self.datasetPath, "train.txt")
        self.text = process_text(self.textPath)
        # with open(textPath, "r", encoding='utf-8') as f:
        #     training_text = len(f.read())
        # self.trainingText = training_text

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        index = idx + 1
        # list_dir =  os.listdir(self.datasetPath)
        mel_name = os.path.join(
            self.datasetPath, "ljspeech-mel-%05d.npy" % index)
        spec_name = os.path.join(
            self.datasetPath, "ljspeech-spec-%05d.npy" % index)
        # print(dir_name)
        character = self.text[idx]
        # print(character)
        character = text_to_sequence(character, [hparams.cleaners])
        character = np.array(character)
        # print("#################")
        mel_np = np.load(mel_name)
        spec_np = np.load(spec_name)

        # # print(mel_np)
        # print(np.shape(mel_np))
        # print(np.shape(spec_np))
        # (time, frequency)

        return {"text": character, "mel": mel_np, "spec": spec_np}


def process_text(train_text_path):
    with open(train_text_path, "r", encoding="utf-8") as f:
        inx = 0
        txt = []
        for line in f.readlines():
            cnt = 0
            for index, ele in enumerate(line):
                if ele == '|':
                    cnt = cnt + 1
                    if cnt == 3:
                        inx = index
                        end = len(line)
                        txt.append(line[inx+1:end-1])
                        break
        return txt


def collate_fn(batch):
    # # Test
    # for d in batch:
    #     # print(d["mel"])
    #     print(np.shape(d["mel"]))

    texts = [d['text'] for d in batch]
    mels = [d['mel'] for d in batch]
    specs = [d['spec'] for d in batch]

    # print(texts)
    texts = pad_seq_text(texts)
    # print(type(texts))
    # print(np.shape(texts))
    # mels = pad_sequence(mels)
    # specs = pad_sequence(specs)
    # print(np.shape(mels[0]))
    # print(np.shape(specs[0]))
    # for mel in mels:
    #     print(np.shape(mel))
    # print()
    mels = pad_seq_spec(mels)
    # print(np.shape(mels))
    specs = pad_seq_spec(specs)

    return {"text": texts, "mel": mels, "spec": specs}


def pad_seq_text(inputs):
    def pad_data(x, length):
        pad = 0
        return np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=pad)

    max_len = max((len(x) for x in inputs))
    return np.stack([pad_data(x, max_len) for x in inputs])


# def pad(x, max_len):
#         # print(type(x))
#     if np.shape(x)[0] > max_len:
#         print("ERROR!")
#     s = np.shape(x)[1]
#     # print(s)
#     x = np.pad(x, (0, max_len - np.shape(x)
#                    [0]), mode='constant', constant_values=0)
#     return x[:, :s]


def pad_seq_spec(inputs):
    def pad(x, max_len):
        # print(type(x))
        if np.shape(x)[0] > max_len:
            # print("ERROR!")
            raise ValueError("not max_len")
        s = np.shape(x)[1]
        # print(s)
        x = np.pad(x, (0, max_len - np.shape(x)
                       [0]), mode='constant', constant_values=0)
        return x[:, :s]

    max_len = max(np.shape(x)[0] for x in inputs)
    # print(max_len)
    # for x in inputs:
    #     x = pad(x, max_len)
    #     # print(np.shape(x))
    #     # print(x)
    # print(np.stack([pad(x,max_len) for x in inputs]))
    # a  = np.stack([pad(x,max_len) for x in inputs])
    # print(np.shape(a))
    # print(type(a))
    return np.stack([pad(x,max_len) for x in inputs])


# def pad_sequence(sequences):
#     '''
#     pad sequence to same length (max length)
#     ------------------
#     input:
#         sequences --- a list of tensor with variable length
#         out --- a tensor with max length
#     '''

#     lengths = [data.size(0) for data in sequences]
#     batch_size = len(sequences)
#     max_len = max(lengths)
#     trailing_dims = sequences[0].size()[1:]
#     out_dims = (batch_size, max_len) + trailing_dims
#     dtype = sequences[0].data.type()
#     out = torch.zeros(*out_dims).type(dtype)

#     for i, data in enumerate(sequences):
#         out[i, :lengths[i]] = data

#     return out


if __name__ == "__main__":
    # seq = np.ndarray([])
    # seq = np.append(seq, np.array([1, 2, 3]))
    # seq = np.append(seq, np.array([1]))
    # seq = np.append(seq, np.array([2, 3]))
    # seq = np.append(seq, [1])
    # seq = np.append(seq, [2, 3])
    # seq = [[[1, 2, 3], [1, 2, 3], [1, 2, 3]]]
    # # seq = torch.Tensor(seq)
    # seq = np.array(seq)
    # print(seq)
    # print(pad_sequence(seq))

    # Test
    dataset = SpeechData("dataset")
    training_loader = DataLoader(
        dataset, batch_size=2, shuffle=True, collate_fn=collate_fn, drop_last=True, num_workers=1)

    for i, data in enumerate(training_loader):
        # print(data)
        a = 0
        print(data)

    # # Other test
    # test_list = [np.array([1, 2, 3]), np.array([1])]
    # # print(text_to_sequence("I like", [hparams.cleaners]))
    # # print(pad_sequence(test_list))

    # # test_list = [np.array([[1, 2, 3], [1, 2, 3]]),
    # #              np.array([[1, 2, 3], [1, 3]])]
    # # print(pad_sequence(test_list))
    # a = np.ndarray((2, 2))
    # print(a)
    # print(pad(a, 3))
