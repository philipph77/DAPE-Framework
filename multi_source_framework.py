import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import numpy as np
import architectures
import datasets

DEBUG_MODE = True

if not(DEBUG_MODE):
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.profiler.emit_nvtx(False)


class Framework(nn.Module):
    def __init__(self, encoders, latent_dim, num_classes, use_adversary=False):
        super().__init__()
        self.num_datasources = len(encoders)
        self.is_trained = False
        self.use_adversary = use_adversary
        self.building_blocks = nn.ModuleList()
        self.building_blocks.extend(encoders)
        if self.use_adversary:
            self.building_blocks.append(architectures.DenseClassifier(latent_dim, self.num_datasources))
        self.building_blocks.append(architectures.DenseClassifier(latent_dim, num_classes))

    def forward(self, x_list):
        if self.use_adversary:
            assert len(self.building_blocks) - 2 ==len(x_list)
        else:
            assert len(self.building_blocks) - 1 == len(x_list)

        z_list = list()
        for datasource_id, x_i in enumerate(x_list):
            z_i = self.building_blocks[datasource_id](x_i)
            z_list.append(z_i)

        # concatenate the outputs
        z = torch.cat(z_list, dim=0)

        # pass z through the classifier
        y_pred = self.building_blocks[-1](z)

        if self.use_adversary:
            d_pred = self.building_blocks[-2](z)
            return y_pred, d_pred

        return y_pred

    def add_encoder(self, encoder):
        # ATTENTION: THIS IS NOT SUPPORTED YET!
        if self.is_trained:
            for param in self.parameters():
                param.requires_grad = False
        self.building_blocks.insert(self.num_datasources, encoder)

    def get_config(self):
        pass

if __name__ == '__main__':
    from helper_funcs import count_parameters
    
    F1, D, F2 = 32, 16, 8
    latent_dim = 20

    encoders = [
        architectures.EEGNetEncoder(channels=32, temporal_filters=F1, spatial_filters=D, pointwise_filters=F2, dropout_propability=0.25, latent_dim=latent_dim, use_constrained_conv=False), #DEAP
        architectures.EEGNetEncoder(channels=14, temporal_filters=F1, spatial_filters=D, pointwise_filters=F2, dropout_propability=0.25, latent_dim=latent_dim, use_constrained_conv=False),  #DREAMER
        architectures.EEGNetEncoder(channels=62, temporal_filters=F1, spatial_filters=D, pointwise_filters=F2, dropout_propability=0.25, latent_dim=latent_dim, use_constrained_conv=False), #SEED
        architectures.EEGNetEncoder(channels=62, temporal_filters=F1, spatial_filters=D, pointwise_filters=F2, dropout_propability=0.25, latent_dim=latent_dim, use_constrained_conv=False) #SEED-IV
    ]
    model = Framework(encoders, 20, 3)

    print("----- #Model Parameters: -----")
    print(count_parameters(model))
    print("---------")
    #summary(model, input_size=[(1,62,2*256),(batch_size,1,62,2*256),(batch_size,1,32,2*128),(batch_size,1,14,2*128)])
    print(model)