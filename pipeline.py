import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import platform
import architectures
from pipeline_funcs import train, train_adversarial, test
from torchinfo import summary
from multi_source_framework import Framework
import datasets


def pipeline():
    if  platform.system() == 'Darwin':
        path = '../../Datasets/private_encs/'
    else:
        path = '../Datasets/private_encs/'
    
    train_datasource_files = [os.path.join(path,'train',f) for f in sorted(os.listdir(os.path.join(path, 'train'))) if f.endswith('.npz') and not('test' in f)]
    validation_datasource_files = [os.path.join(path,'val', f) for f in sorted(os.listdir(os.path.join(path, 'val'))) if f.endswith('.npz') and not('test' in f)]
    test_datasource_files = [os.path.join(path,'test', f) for f in sorted(os.listdir(os.path.join(path, 'test'))) if f.endswith('.npz') and not('test' in f)]

    batch_size = 4*64
    F1, D, F2 = 32, 16, 8
    latent_dim = 20

    encoders = [
        architectures.EEGNetEncoder(channels=32, temporal_filters=F1, spatial_filters=D, pointwise_filters=F2, dropout_propability=0.25, latent_dim=latent_dim, use_constrained_conv=False), #DEAP
        architectures.EEGNetEncoder(channels=14, temporal_filters=F1, spatial_filters=D, pointwise_filters=F2, dropout_propability=0.25, latent_dim=latent_dim, use_constrained_conv=False),  #DREAMER
        architectures.EEGNetEncoder(channels=62, temporal_filters=F1, spatial_filters=D, pointwise_filters=F2, dropout_propability=0.25, latent_dim=latent_dim, use_constrained_conv=False), #SEED
        architectures.EEGNetEncoder(channels=62, temporal_filters=F1, spatial_filters=D, pointwise_filters=F2, dropout_propability=0.25, latent_dim=latent_dim, use_constrained_conv=False) #SEED-IV
    ]
    model = Framework(encoders, 20, 3)

    training_data = datasets.MultiSourceDataset(train_datasource_files)
    validation_data = datasets.MultiSourceDataset(validation_datasource_files)
    batch_sizes = 2**np.arange(4,11)
    batch_sizes = batch_sizes.tolist()
    num_workers = np.arange(1,9).tolist()

    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    validation_dataloader = DataLoader(validation_data, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    if train(model, train_dataloader, validation_dataloader, 'testing_arround', '../logs/', max_epochs=1, early_stopping_after_epochs=1):
        model.is_trained = True

    best_state = torch.load(os.path.join('../logs/', 'testing_arround', 'best_model.pt'))
    model.load_state_dict(best_state['state_dict'])

    test_data = datasets.MultiSourceDataset(test_datasource_files)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    test(model, test_dataloader, 'testing_arround', '../logs')


if __name__ == '__main__':
    pipeline()
    