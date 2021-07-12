from scipy.sparse import data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
import math
import time
from tqdm import trange, tqdm
import architectures
import datasets


class Framework(nn.Module):
    def __init__(self, encoders, latent_dim, num_classes, use_adversary=False):
        super().__init__()
        self.num_datasources = len(encoders)
        self.is_trained = False
        self.use_adversary = False
        self.building_blocks = nn.ModuleList()
        self.building_blocks.extend(encoders)
        self.building_blocks.append(architectures.DenseClassifier(latent_dim, num_classes))

    def forward(self, x_list):
        assert len(self.building_blocks) - 1 == len(x_list)

        z_list = list()
        for datasource_id, x_i in enumerate(x_list):
            z_i = self.building_blocks[datasource_id](x_i)
            z_list.append(z_i)

        # concatenate the outputs
        z = torch.cat(z_list, dim=0)

        # pass z through the classifier
        y_pred = self.building_blocks[-1](z)

        return y_pred

    def add_datasource(self, encoder, x, d):
        # ATTENTION: THIS IS NOT SUPPORTED YET!
        if not(self.is_trained):
            print("The Framework was not trained yet, you can add the data-source be training the whole framework")
            return
        for param in self.parameters():
            param.requires_grad = False
        self.building_blocks.insert(-1,encoder)
        #TODO: insert training procedure for the encoder here
        pass

def train(model, datasource_files, max_epochs=500, batch_size=64):
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    training_data = datasets.MultiSourceDataset(datasource_files)
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    validation_data = datasets.MultiSourceDataset(datasource_files)
    validation_dataloader = DataLoader(validation_data, batch_size=batch_size, shuffle=True)

    for epoch in range(max_epochs):
        model.train()
        start_time = time.time()
        loss = 0.
        optimizer.zero_grad()
        for x_list,y_true in tqdm(train_dataloader):
            x_list = [x_i.to(device) for x_i in x_list]
            y_true = y_true.to(device)
            y_true = torch.flatten(y_true)
            y_pred = model(x_list)
            y_pred.squeeze_()
            loss = criterion(y_pred, y_true)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            acc = 0.
            for x_val_list, y_true in tqdm(validation_dataloader):
                x_val_list = [x_i.to(device) for x_i in x_val_list]
                y_true = y_true.to(device)
                y_true = torch.flatten(y_true)
                y_val_pred = model(x_val_list)
                y_val_pred.squeeze_()
                acc += accuracy_score(y_true.detach().cpu().numpy(), np.argmax(y_val_pred.detach().cpu().numpy(), axis=1))
        
        acc = acc / len(validation_data)
        end_time = time.time()
        print("Epoch %i: - Loss: %4.2f - Accuracy: %4.2f - Elapsed Time: %4.2f s"%(epoch, loss, acc, end_time-start_time))

if __name__ == '__main__':
    import os
    from torchinfo import summary
    import platform
    from sklearn.model_selection import train_test_split
    import helper_funcs

    if  platform.system() == 'Darwin':
        path = '../../Datasets/private_encs/'
    else:
        path = '../Datasets/private_encs/'
    datasource_files = [os.path.join(path,f) for f in sorted(os.listdir(path)) if f.endswith('.npz') and 'split' in f and not('test' in f)]

    batch_size = 64
    F1, D, F2 = 32, 16, 8

    encoders = [
        architectures.EEGNetEncoder(channels=32, temporal_filters=F1, spatial_filters=D, pointwise_filters=F2, dropout_propability=0.25, latent_dim=20), #DEAP
        architectures.EEGNetEncoder(channels=14, temporal_filters=F1, spatial_filters=D, pointwise_filters=F2, dropout_propability=0.25, latent_dim=20),  #DREAMER
        architectures.EEGNetEncoder(channels=62, temporal_filters=F1, spatial_filters=D, pointwise_filters=F2, dropout_propability=0.25, latent_dim=20), #SEED
        architectures.EEGNetEncoder(channels=62, temporal_filters=F1, spatial_filters=D, pointwise_filters=F2, dropout_propability=0.25, latent_dim=20) #SEED-IV
    ]
    model = Framework(encoders, 20, 3)

    print("----- #Model Parameters: -----")
    print(helper_funcs.count_parameters(model))
    print("---------")
    #summary(model, input_size=[(1,62,2*256),(batch_size,1,62,2*256),(batch_size,1,32,2*128),(batch_size,1,14,2*128)])
    #print(model)

    train(model, datasource_files)
    