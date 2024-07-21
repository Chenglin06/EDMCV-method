import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
import numpy as np


def load_data(batch_size, num_worker):
    all_name = []
    all_label = []
    all_data = []
    
    #esm后的编码encode_data_2570
    for lin in open('../../esm_encode/encode_data_1708','r',encoding = 'utf-8'):
        name,label,data = lin.strip().split('\t')
        data = eval(data)
        all_name.append(name)
        all_label.append(int(label))
        all_data.append(data)
    #print(len(all_data))
    
    
    all_data = np.array(all_data, dtype=np.float64)
    all_label = np.array(all_label)
    #all_data = all_data.float()
    train_dataset = CustomDataset(torch.tensor(all_data), torch.tensor(all_label))
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,num_workers=num_worker, shuffle=False)
    
    
    return train_dataloader, train_dataloader
    #print(len(train_dataloader))
    #exit()
    #test_dataloader = DataLoader(test_dataset, batch_size=batch_size,num_workers=2, shuffle=False)
    
    #test_dataset = CustomDataset(torch.tensor(get_data.test_x), torch.tensor(get_data.test_y))
        #print(name)
        #exit()
                

class CustomDataset(Dataset):
    def __init__(self, sequences1, labels):
        self.sequences1 = sequences1
        #self.sequences2 = sequences2
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        sequence1 = self.sequences1[index]
        #sequence2 = self.sequences2[index]
        label = self.labels[index]
        
        return sequence1, label
