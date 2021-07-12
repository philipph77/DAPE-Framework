from scipy.sparse import data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
import math
import time
from tqdm import trange
import architectures


class Framework(nn.Module):
    def __init__(self, encoders, latent_dim, num_classes):
        super().__init__()
        self.num_datasources = len(encoders)
        self.is_trained = False
        self.encoders = encoders
        self.emotion_classifier = architectures.DenseClassifier(latent_dim, num_classes)
        self.params = list()
        for i in range(self.num_datasources):
            self.params.append({'params': self.encoders[i].parameters()})
        self.params.append({'params': self.emotion_classifier.parameters()})


    def forward(self, x_list, y_batch):
        assert len(self.encoders) == len(x_list)

        z_list = list()
        for datasource_id, x_i in enumerate(x_list):
            z_i = self.encoders[datasource_id](x_i)
            z_list.append(z_i)

        # concatenate the outputs
        z = torch.cat(z_list, dim=0)

        # shuffle
        z, y_batch = shuffle(z, y_batch)

        # pass z through the classifier
        y_pred = self.emotion_classifier(z)

        return y_pred, y_batch

    def add_datasource(self, encoder, x, d):
        if not(self.is_trained):
            print("The Framework was not trained yet, you can add the data-source be training the whole framework using your data-source (among other)")
            return
        pass

def train(model, x_train_list, y_train_list, x_val_list, y_val_list, max_epochs=500, batch_size=32):
    optimizer = optim.Adam(model.params, lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    y_val = torch.cat(y_val_list, dim=0)
    for i in range(len(x_val_list)):
        x_val_list[i].unsqueeze_(1)

    # prepare the batches
    batch_start_idx = list()
    batch_end_idx = list()
    num_batches = math.ceil(float(x_train_list[0].shape[0]) / batch_size)
    # note: it is necessary, that each encoder recieves the exact same amount of batches
    for datasource_id in range(model.num_datasources):
        assert math.ceil(float(x_train_list[datasource_id].shape[0]) / batch_size) == num_batches
    batch_start_idx = np.arange(0, num_batches*batch_size, batch_size)
    batch_end_idx = batch_start_idx + batch_size
    batch_end_idx[-1] = x_train_list[0].shape[0]
    assert len(batch_start_idx) == num_batches
    assert len(batch_end_idx) == num_batches

    # pass the batches to the forward method (as a list)
    for epoch in range(max_epochs):
        start_time = time.time()
        acc = 0.
        loss = 0.
        for batch_idx in trange(num_batches):
            x_batch_list = list()
            y_batch_list = list()
            optimizer.zero_grad()
            for i in range(model.num_datasources):
                x_batch_list.append(x_train_list[i][batch_start_idx[batch_idx]:batch_end_idx[batch_idx],:,:].unsqueeze_(1))
                y_batch_list.append(y_train_list[i][batch_start_idx[batch_idx]:batch_end_idx[batch_idx]])
            y_batch = torch.cat(y_batch_list, dim=0)
            y_pred, y_batch = model(x_batch_list, y_batch)
            y_pred.squeeze_()
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            #acc = acc + accuracy_score(y_batch.detach().numpy(), np.argmax(y_pred.detach().numpy(),axis=1))
        # This line is to handle the fact that the last batch may be smaller then the other batches
        #acc = acc - accuracy_score(y_batch.detach().numpy(), np.argmax(y_pred.detach().numpy(),axis=1)) + float(batch_end_idx[-1]-batch_start_idx[-1])/batch_size  * accuracy_score(y_batch.detach().numpy(), np.argmax(y_pred.detach().numpy(),axis=1))

        #acc = acc / (float(x_train_list[0].shape[0]) / batch_size)
        
        end_time = time.time()

        y_val_pred, y_val = model(x_val_list, y_val)
        y_val_pred.squeeze_()

        acc = accuracy_score(y_val.detach().numpy(), np.argmax(y_val_pred.detach().numpy(), axis=1))

        print("Epoch %i: - Loss: %4.2f - Accuracy: %4.2f - Elapsed Time: %4.2f s"%(epoch, loss, acc, end_time-start_time))

if __name__ == '__main__':
    import os
    from torchinfo import summary
    import platform
    from sklearn.model_selection import train_test_split
    C = 32
    T = 2*128
    F1 = 32
    D = 16
    F2 = 8

    encoders = {
        0: architectures.EEGNetEncoder(channels=62, temporal_filters=F1, spatial_filters=D, pointwise_filters=F2, dropout_propability=0.25, latent_dim=20), #SEED
        1: architectures.EEGNetEncoder(channels=62, temporal_filters=F1, spatial_filters=D, pointwise_filters=F2, dropout_propability=0.25, latent_dim=20), #SEED-IV
        2: architectures.EEGNetEncoder(channels=32, temporal_filters=F1, spatial_filters=D, pointwise_filters=F2, dropout_propability=0.25, latent_dim=20), #DEAP
        3: architectures.EEGNetEncoder(channels=14, temporal_filters=F1, spatial_filters=D, pointwise_filters=F2, dropout_propability=0.25, latent_dim=20)  #DREAMER
    }

    batch_size = 64
    num_batches = 1000

    model = Framework(encoders, 20, 3)

    
    #summary(model, input_size=[(batch_size*num_batches,1,C,T),(batch_size*num_batches,1,C,T),(batch_size*num_batches,1,C,T),(batch_size*num_batches,1,C,T)])

    if  platform.system() == 'Darwin':
        path = '../../Datasets/private_encs/'
    else:
        path = '../Datasets/private_encs/'

    
    x_list = list()
    y_list = list()

    datasource_files = [f for f in os.listdir(path) if f.endswith('.npz') and 'split' in f and not('test' in f)]

    for datasource_file in datasource_files:
        datasource = np.load(os.path.join(path, datasource_file))
        x_list.append(torch.from_numpy(datasource['X']).type(torch.FloatTensor))
        y_list.append(torch.from_numpy(datasource['Y']).type(torch.LongTensor) + 1)

    x_train_list = list()
    x_val_list = list()
    x_test_list = list()

    y_train_list = list()
    y_val_list = list()
    y_test_list = list()

    for i in range(len(x_list)):
        x_train_element, x_test_element, y_train_element, y_test_element = train_test_split(x_list[i], y_list[i], train_size=0.6)
        x_val_element, x_test_element, y_val_element, y_test_element = train_test_split(x_test_element, y_test_element, test_size=0.5)
        x_train_list.append(x_test_element)
        x_val_list.append(x_val_element)
        x_test_list.append(x_test_element)
        y_train_list.append(y_train_element)
        y_val_list.append(y_val_element)
        y_test_list.append(y_test_element)

    train(model, x_train_list, y_train_list, x_val_list, y_val_list)
    