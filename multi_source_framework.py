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
    
    def custom_train(self, x_list, y_list, max_epochs=500, batch_size=64):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        x_list, y_list = x_list.to(device), y_list.to(device)

        optimizer = optim.Adam(self.params, lr=1e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        # prepare the batches
        batch_start_idx = list()
        batch_end_idx = list()
        num_batches = math.ceil(float(x_list[0].shape[0]) / batch_size)
        # note: it is necessary, that each encoder recieves the exact same amount of batches
        for datasource_id in range(self.num_datasources):
            assert math.ceil(float(x_list[datasource_id].shape[0]) / batch_size) == num_batches
        batch_start_idx = np.arange(0, num_batches*batch_size, batch_size)
        batch_end_idx = batch_start_idx + batch_size
        batch_end_idx[-1] = x_list[0].shape[0]
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
                for i in range(self.num_datasources):
                    x_batch_list.append(x_list[i][batch_start_idx[batch_idx]:batch_end_idx[batch_idx],:,:].unsqueeze_(1))
                    y_batch_list.append(y_list[i][batch_start_idx[batch_idx]:batch_end_idx[batch_idx]])
                y_batch = torch.cat(y_batch_list, dim=0)
                y_pred, y_batch = self.forward(x_batch_list, y_batch)
                y_pred.squeeze_()
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                acc = acc + accuracy_score(y_batch.detach().numpy(), np.argmax(y_pred.detach().numpy(),axis=1))
            # This line is to handle the fact that the last batch may be smaller then the other batches
            acc = acc - accuracy_score(y_batch.detach().numpy(), np.argmax(y_pred.detach().numpy(),axis=1)) + float(batch_end_idx[-1]-batch_start_idx[-1])/batch_size  * accuracy_score(y_batch.detach().numpy(), np.argmax(y_pred.detach().numpy(),axis=1))

            acc = acc / (float(x_list[0].shape[0]) / batch_size)
            
            end_time = time.time()

            print("Epoch %i: - Loss: %4.2f - Accuracy: %4.2f - Elapsed Time: %4.2f s"%(epoch, loss, acc, end_time-start_time))


    def add_datasource(self, encoder, x, d):
        if not(self.is_trained):
            print("The Framework was not trained yet, you can add the data-source be training the whole framework using your data-source (among other)")
            return
        pass


if __name__ == '__main__':
    from torchinfo import summary
    import platform
    C = 32
    T = 2*128
    F1 = 32
    D = 16
    F2 = 8

    encoders = {
        0: architectures.EEGNetEncoder(channels=C, temporal_filters=F1, spatial_filters=D, pointwise_filters=F2, dropout_propability=0.25, latent_dim=20),
        1: architectures.EEGNetEncoder(channels=C, temporal_filters=F1, spatial_filters=D, pointwise_filters=F2, dropout_propability=0.25, latent_dim=20),
        2: architectures.EEGNetEncoder(channels=C, temporal_filters=F1, spatial_filters=D, pointwise_filters=F2, dropout_propability=0.25, latent_dim=20),
        3: architectures.EEGNetEncoder(channels=C, temporal_filters=F1, spatial_filters=D, pointwise_filters=F2, dropout_propability=0.25, latent_dim=20)
    }

    batch_size = 64
    num_batches = 1000

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Framework(encoders, 20, 3)

    model = model.to(device)
    #summary(model, input_size=[(batch_size*num_batches,1,C,T),(batch_size*num_batches,1,C,T),(batch_size*num_batches,1,C,T),(batch_size*num_batches,1,C,T)])

    if  platform.system() == 'Darwin':
        dataset = np.load('../../Datasets/private_encs/DEAP.npz')
    else:
        dataset = np.load('../Datasets/private_encs/DEAP.npz')
    x_list = list()
    y_list = list()

    for i in range(len(encoders)):
        x_list.append(torch.from_numpy(dataset['X']).type(torch.FloatTensor))
        y_list.append(torch.from_numpy(dataset['Y']).type(torch.LongTensor) + 1)

    #model.custom_train(x_list,y_list)
    