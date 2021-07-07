import torch
import torch.nn as nn
import torch.nn.functional as F
import architectures

class Framework(nn.Module):
    def __init__(self, num_data_sources, latent_dim, num_classes, encoder_backbone):
        super().__init__()
        encoder_list = list()
        for i in range(num_data_sources):
            encoder_list.append(encoder_backbone)
        emotion_classifier = architectures.DenseClassifier(latent_dim, num_classes)

    def forward(self, x):
        return x


if __name__ == '__main__':
    C = 62
    T = 128
    F1 = 32
    D = 16
    F2 = 8
    encoder = architectures.EEGNetEncoder(channels=C, temporal_filters=F1, spatial_filters=D, pointwise_filters=F2, dropout_propability=0.25, latent_dim=20)