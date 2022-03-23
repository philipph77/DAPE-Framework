import torch
import torch.nn as nn
from torch.nn.functional import normalize
from torch.utils.data import DataLoader
import os
import numpy as np
from tqdm.utils import _environ_cols_wrapper
import architectures
import datasets

DEBUG_MODE = True

if not(DEBUG_MODE):
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.profiler.emit_nvtx(False)


class autoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder  = nn.ModuleList(encoder)
        self.decoder = nn.ModuleList(decoder)

    def forward(self,x):
        z_pred = self.encoder[0](x)
        x_pred = self.decoder[0](z_pred)
        return x_pred, z_pred


if __name__ == '__main__':
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import architectures
    from autoencoder import autoencoder
    from datasets import SingleSourceDataset, MultiSourceDataset
    from torch.utils.data import DataLoader
    from hyperparam_schedulers import constant_schedule
    from tqdm import trange, tqdm

    def draw_figure(x, x_pred, num_channels=5):
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        color_index = np.linspace(0,1, num_channels)
        fig = plt.figure(figsize=(10, 5))
        gs = GridSpec(nrows=2, ncols=1, height_ratios=[1, 1])
        ax0 = fig.add_subplot(gs[0,:])
        ax1 = fig.add_subplot(gs[1,:])
        plt.ion()
        plt.show()
        for i in range(num_channels):
            ax0.plot(x[0,0,i,:], label=i, color=plt.cm.RdYlBu(color_index[i]))
            ax1.plot(x_pred[0,0,i,:], label=i, color=plt.cm.RdYlBu(color_index[i]))
        ax0.set_title('True results')
        ax1.set_title('Predicted results')
        ax0.legend()
        plt.draw()
        plt.pause(0.001)


    dataset_name = 'DEAP'
    latent_dim = 1000
    batch_size = 64
    epochs = 200

    if dataset_name == 'SEED' or dataset_name == 'SEED_IV':
        C, T = 62, 400
    elif dataset_name  == 'DEAP':
        C,T = 32, 256
    elif dataset_name == 'DREAMER':
        C,T = 14,256
    else:
        raise NotImplementedError

    encoder = architectures.DeepConvNetEncoder(C, latent_dim)
    decoder = architectures.DeepConvNetDecoder(latent_dim, T, C)

    encoder = architectures.VanillaEncoder(C, T, latent_dim)
    decoder = architectures.VanillaDecoder(C, T, latent_dim)

    model = autoencoder(encoder, decoder)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_loss_logs = list()
    val_loss_logs = list()

    training_data = SingleSourceDataset('../../Datasets/private_encs/train/class_stratified_%s.npz'%dataset_name, supress_output=True, normalize=False)
    validation_data = SingleSourceDataset('../../Datasets/private_encs/val/class_stratified_%s.npz'%dataset_name, supress_output=True, normalize=False)

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(validation_data, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.L1Loss()


    model = model.to(device)
    for epoch in trange(1,epochs+1, desc='Training Progess'):
        train_loss = 0
        val_loss = 0
        model.train()
        for x, _ in tqdm(train_dataloader, desc='Epoch: %i'%epoch, leave=False):
            #optimizer.zero_grad()
            for param in model.parameters():
                param.grad = None
            x = x.to(device)
            x_pred, _ = model(x)
            loss = criterion(x_pred, x)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        train_loss_logs.append(train_loss/len(train_dataloader))
        model.eval()
        with torch.no_grad():
            for x, _ in val_dataloader:
                x = x.to(device)
                x_pred, _ = model(x)
                val_loss += criterion(x_pred, x).item()
            val_loss_logs.append(val_loss / len(val_dataloader))
        #draw_figure(x, x_pred)
        print("Train-Loss: %4.4f - Validation-Loss: %4.4f"%(train_loss_logs[-1], val_loss_logs[-1]))