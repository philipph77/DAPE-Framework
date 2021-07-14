import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

class MultiSourceDataset(Dataset):
    def __init__(self, datasource_files):
        self.num_datasources = len(datasource_files)
        self.x_list = list()
        self.y_list = list()
        for datasource_file in datasource_files:
            print("Loading %s"%datasource_file)
            datasource = np.load(datasource_file)
            self.x_list.append(torch.from_numpy(datasource['X']).type(torch.FloatTensor))
            self.y_list.append(torch.from_numpy(datasource['Y']).type(torch.LongTensor) + 1)
        self.length = self.x_list[0].shape[0]
        for i in range(self.num_datasources):
            assert self.length == self.x_list[0].shape[0]


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x_item = list()
        y_item = list()
        d_item = list()
        for i in range(self.num_datasources):
            x_item.append(self.x_list[i][idx,:,:].unsqueeze_(0))
            y_item.append(self.y_list[i][idx].reshape(-1,1))
            d_item.append(torch.tensor(i).reshape(-1,1))
        y_item = torch.cat(y_item, dim=0).squeeze()
        d_item = torch.cat(d_item, dim=0).squeeze()
        return x_item, y_item, d_item


def test_dimensions():
    import platform
    from torch.utils.data import DataLoader

    if  platform.system() == 'Darwin':
        path = '../../Datasets/private_encs/'
    else:
        path = '../Datasets/private_encs/'

    datasource_files = [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.npz') and 'split' in f and not('test' in f)]
    training_data = MultiSourceDataset(datasource_files)
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    for x_batch_list,y_batch, d_batch in train_dataloader:
        y_batch = torch.flatten(y_batch)
        d_batch = torch.flatten(d_batch)
        print("X: %s - Y: %s - D: %s"%(str(x_batch_list[0].shape), str(y_batch.shape), str(d_batch.shape)))

if __name__ == '__main__':
    test_dimensions()