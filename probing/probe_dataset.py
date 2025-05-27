from torch.utils.data import Dataset
import torch
import numpy as np

class ProbeDataset(Dataset):
    def __init__(self, encodings, labels, label2id):
        self.encodings  = encodings
        self.labels = labels
        self.label2id = label2id

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        encoding = self.encodings[index]
        label = self.labels[index]
        encoding = torch.from_numpy(encoding).float()
        label = torch.from_numpy(np.array(label)).long()
        return encoding, label