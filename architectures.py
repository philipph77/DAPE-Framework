import layers
import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGNetEncoder(nn.Module):
    def __init__(self, channels, temporal_filters, spatial_filters, pointwise_filters, dropout_propability, latent_dim):
        """ For details about the Parameters see
        Lawhern, Vernon J., et al. "EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces." Journal of neural engineering 15.5 (2018): 056013.

        """

        super().__init__()

        # Block 1
        self.conv1 = nn.Conv2d(1,temporal_filters,(1,64), padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(temporal_filters)
        self.conv2 = layers.ConstrainedConv2d(temporal_filters, spatial_filters*temporal_filters, kernel_size=(channels,1), padding='valid', groups=temporal_filters, bias=False) # DepthwiseConv2d
        #self.conv2 = nn.Conv2d(temporal_filters, spatial_filters*temporal_filters, kernel_size=(channels,1), padding='valid', groups=temporal_filters, bias=False) # DepthwiseConv2d)
        self.bn2 = nn.BatchNorm2d(spatial_filters*temporal_filters)
        self.act1 = nn.ELU()
        self.pool1 = nn.AvgPool2d((1,4))
        self.drop1 = nn.Dropout(dropout_propability)

        # Block 2
        self.conv3_0 = nn.Conv2d(spatial_filters*temporal_filters, spatial_filters*temporal_filters, kernel_size=(1,16), padding='same', groups=spatial_filters*temporal_filters, bias=False) # SeparableConv2d Part1
        self.conv3_1 = nn.Conv2d(spatial_filters*temporal_filters, pointwise_filters, kernel_size=1, padding='same', bias=False) # SeparableConv2d Part2
        self.bn3 = nn.BatchNorm2d(pointwise_filters)
        self.act2 = nn.ELU()
        self.pool2 = nn.AvgPool2d((1,8))
        self.drop2 = nn.Dropout(dropout_propability)

        self.flat1 = nn.Flatten()
        self.pool3 = nn.AdaptiveAvgPool1d(latent_dim)
        

    def forward(self, x):
        #x.unsqueeze_(1)

        # Block 1
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.conv2(h)
        h = self.bn2(h)
        h = self.act1(h)
        h = self.pool1(h)
        h = self.drop1(h)

        # Block 2
        h = self.conv3_0(h)
        h = self.conv3_1(h)
        h = self.bn3(h)
        h = self.act2(h)
        h = self.pool2(h)
        h = self.drop2(h)

        h = self.flat1(h)
        h.unsqueeze_(1)
        z = self.pool3(h)

        return z

class DeepConvNetEncoder(nn.Module):
    def __init__(self, channels, latent_dim):
        super().__init__()

        # Block 1
        self.conv1_0 = nn.Conv2d(1, 25, (1, 5) ) 
        self.conv1_1 = nn.Conv2d(25, 25, (channels,1), bias=False)
        self.bn1 = nn.BatchNorm2d(25, eps=1e-05, momentum=0.1)
        self.act1 = nn.ELU()
        self.pool1 = nn.MaxPool2d((1,2), stride=(1,2))
        self.drop1 = nn.Dropout(0.5)

        # Block 2
        self.conv2 = nn.Conv2d(25, 50, (1,5), bias=False)
        self.bn2 = nn.BatchNorm2d(50, eps=1e-05, momentum=0.1)
        self.act2 = nn.ELU()
        self.pool2 = nn.MaxPool2d((1,2), stride=(1,2))
        self.drop2 = nn.Dropout(0.5)

        # Block 3
        self.conv3 = nn.Conv2d(50, 100, (1,5), bias=False)
        self. bn3 = nn.BatchNorm2d(100, eps=1e-05, momentum=0.1)
        self.act3 = nn.ELU()
        self.pool3 = nn.MaxPool2d((1,2), stride=(1,2))
        self.drop3 = nn.Dropout(0.5)

        # Block 4
        self.conv4 = nn.Conv2d(100, 200, (1,5), bias=False)
        self. bn4 = nn.BatchNorm2d(200, eps=1e-05, momentum=0.1)
        self.act4 = nn.ELU()
        self.pool4 = nn.MaxPool2d((1,2), stride=(1,2))
        self.drop4 = nn.Dropout(0.5)

        # towards classifier
        self.flat = nn.Flatten()
        self.pool5 = nn.AdaptiveAvgPool1d(latent_dim)

    def forward(self, x):
        # Block 1
        h = self.conv1_0(x)
        h = self.conv1_1(h)
        h = self.bn1(h)
        h = self.act1(h)
        h = self.pool1(h)
        h = self.drop1(h)
        
        # Block 2
        h = self.conv2(h)
        h = self.bn2(h)
        h = self.act2(h)
        h = self.pool2(h)
        h = self.drop2(h)

        # Block 3
        h = self.conv3(h)
        h = self.bn3(h)
        h = self.act3(h)
        h = self.pool3(h)
        h = self.drop3(h)

        # Block 4
        h = self.conv4(h)
        h = self.bn4(h)
        h = self.act4(h)
        h = self.pool4(h)
        h = self.drop4(h)

        # towards classifier
        h = self.flat(h)
        h.unsqueeze_(1)
        z = self.pool5(h)

        return z

class ShallowConvNetEncoder(nn.Module):
    def __init__(self, channels, latent_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 40, (1,13))
        self.conv2 = nn.Conv2d(40, 40,(channels, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(eps=1e-05, momentum=0.1)
        self.pool1 = nn.AvgPool2d((1,35), (1,7))
        self.drop1 = nn.Dropout(0.5)
        self.flat = nn.Flatten()
        self.pool2 = nn.AdaptiveAvgPool1d(latent_dim)

    def forward(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.bn1(h)
        h = torch.square(h)
        h = self.pool1(h)
        h = torch.log(torch.clip(h, 1e-7, 10000))
        h = self.drop1(h)
        h = self.flat(h)
        y = self.pool2(h)

        return y

class MLPEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        pass

class DenseClassifier(nn.Module):
    def __init__(self, latent_dim, classes, max_norm=0.25, use_bias=False):
        super().__init__()
        if max_norm:
            self.fc1 = layers.ConstrainedLinear(latent_dim, classes, bias=use_bias)
        else:
            self.fc1 = nn.Linear(latent_dim, classes, bias=use_bias)

    def forward(self, x):
        return self.fc1(x)


if __name__ == '__main__':
    from torchsummary import summary
    C = 62
    T = 128
    F1 = 32
    D = 16
    F2 = 8
    model = EEGNetEncoder(channels=C, temporal_filters=F1, spatial_filters=D, pointwise_filters=F2, dropout_propability=0.25, latent_dim=20)
    summary(model, (1,C,T), batch_size=32)

    #cla = DenseClassifier(20,3)
    #summary(cla, (1,20))