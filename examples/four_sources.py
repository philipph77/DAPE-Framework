import sys
sys.path.append("../eeg_private_encoders")
from multi_source_framework import Framework
import architectures
import datasets
from pipeline_funcs import train, train_adversarial
from torch.utils.data import DataLoader
import platform
import os

def run_four_sources():
    """
    runs an exemplary DAPE Framework on four datasources
    """    
    
    if  platform.system() == 'Darwin':
        path = '../../Datasets/private_encs/'
    else:
        path = '../Datasets/private_encs/'
    
    train_datasource_files = [os.path.join(path,'train',f) for f in sorted(os.listdir(os.path.join(path, 'train'))) if f.endswith('.npz') and not('test' in f)]
    validation_datasource_files = [os.path.join(path,'val', f) for f in sorted(os.listdir(os.path.join(path, 'val'))) if f.endswith('.npz') and not('test' in f)]

    latent_dim = 2000
    encoders = [
        architectures.DeepConvNetEncoder(channels=32, latent_dim=latent_dim), #DEAP
        architectures.DeepConvNetEncoder(channels=14, latent_dim=latent_dim), #DREAMER
        architectures.DeepConvNetEncoder(channels=62, latent_dim=latent_dim), #SEED
        architectures.DeepConvNetEncoder(channels=62, latent_dim=latent_dim) #SEED-IV
    ]
    
    training_data = datasets.MultiSourceDataset(train_datasource_files)
    validation_data = datasets.MultiSourceDataset(validation_datasource_files)

    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    validation_dataloader = DataLoader(validation_data, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    model = Framework(encoders, latent_dim, 3, use_adversary=False)
    train(model, train_dataloader, validation_dataloader, 'all-DCN-2000', '../logs/', max_epochs=300)

    model_adv = Framework(encoders, latent_dim, 3, use_adversary=True)
    train_adversarial(model_adv, train_dataloader, validation_dataloader, 'all-DCN-2000-adv', '../logs/',  0.05, max_epochs=300)      

if __name__ == '__main__':
    run_four_sources()