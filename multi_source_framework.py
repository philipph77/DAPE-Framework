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
        self.encoders = nn.ModuleList()
        self.encoders.extend(encoders)
        self.emotion_classifier = nn.ModuleList()
        self.emotion_classifier.append(architectures.DenseClassifier(latent_dim, num_classes)) # Emotion Classifier
        if self.use_adversary:
            self.domain_classifier = nn.ModuleList()
            self.domain_classifier.append(architectures.DenseClassifier(latent_dim, self.num_datasources)) # Adversary
        self.individual_classifiers = nn.ModuleList()
        for _ in range(len(encoders)):
            self.individual_classifiers.append(architectures.DenseClassifier(latent_dim, num_classes)) # Individual Classifiers

    def forward(self, x_list, individual_outputs=False, output_latent_representation=False):
        assert len(self.encoders) ==len(x_list)

        z_list = list()
        for datasource_id, x_i in enumerate(x_list):
            z_i = self.encoders[datasource_id](x_i)
            z_list.append(z_i)
        
        # pass z through the individual classifiers
        y_pred_individual_list = list()
        for datasource_id, z_i in enumerate(z_list):
            y_pred_individual = self.individual_classifiers[datasource_id](z_i)
            y_pred_individual_list.append(y_pred_individual)
        
        # concatenate the latent representation
        z = torch.cat(z_list, dim=0)

        # pass z through the classifier
        y_pred = self.emotion_classifier[0](z)

        return_value = y_pred

        if self.use_adversary:
            d_pred = self.domain_classifier[0](z)
            return_value = return_value, d_pred
        
        if individual_outputs:
            return_value = return_value, y_pred_individual_list
        
        if output_latent_representation:
            return_value = return_value, z_list

        return return_value

    def add_encoder(self, encoder):
        # ATTENTION: THIS IS NOT SUPPORTED YET!
        if self.is_trained:
            for param in self.parameters():
                param.requires_grad = False
        self.encoders.append(encoder)

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