import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import platform
import architectures
from pipeline_funcs import train, train_adversarial, test, test_adversarial
from torchinfo import summary
from multi_source_framework import Framework
import datasets
import pipeline_helper

def pipeline(data_sources, encoder ,latent_dim, adversarial, run_name, lam=0., **kwargs):
    if  platform.system() == 'Darwin':
        path = '../../Datasets/private_encs/'
    else:
        path = '../Datasets/private_encs/'
    
    train_datasource_files = [os.path.join(path,'train',f) for f in sorted(os.listdir(os.path.join(path, 'train'))) if f.endswith('.npz') and not('test' in f)]
    validation_datasource_files = [os.path.join(path,'val', f) for f in sorted(os.listdir(os.path.join(path, 'val'))) if f.endswith('.npz') and not('test' in f)]
    test_datasource_files = [os.path.join(path,'test', f) for f in sorted(os.listdir(os.path.join(path, 'test'))) if f.endswith('.npz') and not('test' in f)]

    # Select only relevant datasource_files    
    data_sources = sorted(data_sources)
    train_datasource_files = pipeline_helper.filter_datasource_files(train_datasource_files, data_sources)
    validation_datasource_files = pipeline_helper.filter_datasource_files(train_datasource_files, data_sources)
    test_datasource_files = pipeline_helper.filter_datasource_files(test_datasource_files, data_sources)

    # build the encoder list
    encoders = pipeline_helper.generate_encoder_list(encoder, latent_dim, test_datasource_files, **kwargs)

    model = Framework(encoders, latent_dim, 3, adversarial)

    training_data = datasets.MultiSourceDataset(train_datasource_files)
    validation_data = datasets.MultiSourceDataset(validation_datasource_files)

    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    validation_dataloader = DataLoader(validation_data, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    if adversarial:
        train_adversarial(model, train_dataloader, validation_dataloader, run_name, '../logs/', lam, max_epochs=300)
    else:
        train(model, train_dataloader, validation_dataloader, run_name, '../logs/', max_epochs=300)

    best_state = torch.load(os.path.join('../logs/', run_name, 'best_model.pt'))
    model.load_state_dict(best_state['state_dict'])

    test_data = datasets.MultiSourceDataset(test_datasource_files)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    if adversarial:
        test_adversarial(model, test_dataloader, run_name, '../logs/')
    else:
        test(model, test_dataloader, run_name, '../logs/')

def pipeline_saverun(data_sources, encoder ,latent_dim, adversarial, run_name, lam=0., **kwargs):
    try:
        pipeline(data_sources, encoder ,latent_dim, adversarial, run_name, lam, **kwargs)
    except Exception as e:
        pipeline_helper.send_mail_notification('Fehler', run_name)
        print(e)

if __name__ == '__main__':
    F1, D, F2, dropout_p = 32, 16, 8, 0.5
    eeg_encoder_args = F1, D, F2

    '''
    pipeline(['SEED'], architectures.EEGNetEncoder, 2000, False, 'EEG-1000-2000-noa', temporal_filters=F1, spatial_filters=D, pointwise_filters=F2, dropout_propability=0.5)
    pipeline_saverun(['SEED'], architectures.EEGNetEncoder, 1000, False, 'EEG-1000-1000-noa', temporal_filters=F1, spatial_filters=D, pointwise_filters=F2, dropout_propability=0.5)
    pipeline_saverun(['SEED'], architectures.EEGNetEncoder, 1000, True, 'EEG-1000-1000-adv-005', temporal_filters=F1, spatial_filters=D, pointwise_filters=F2, dropout_propability=0.5)

    pipeline_saverun(['SEED'], architectures.DeepConvNetEncoder, 1000, False, 'DCN-1000-1000-noa')
    pipeline_saverun(['SEED'], architectures.DeepConvNetEncoder, 1000, True, 'DCN-1000-1000-adv-005')

    pipeline_saverun(['SEED'], architectures.DeepConvNetEncoder, 2000, False, 'DCN-1000-2000-noa')
    pipeline_saverun(['SEED_IV'], architectures.DeepConvNetEncoder, 2000, False, 'DCN-0100-2000-noa')
    pipeline_saverun(['DEAP'], architectures.DeepConvNetEncoder, 2000, False, 'DCN-0010-2000-noa')
    pipeline_saverun(['DREAMER'], architectures.DeepConvNetEncoder, 2000, False, 'DCN-0001-2000-noa')

    pipeline_saverun(['SEED', 'SEED_IV'], architectures.DeepConvNetEncoder, 1000, False, 'DCN-1100-1000-noa')
    pipeline_saverun(['SEED', 'SEED_IV', 'DEAP', 'DREAMER'], architectures.DeepConvNetEncoder, 1000, False, 'DCN-1111-1000-noa')

    pipeline_saverun(['SEED', 'SEED_IV'], architectures.DeepConvNetEncoder, 1000, True, 'DCN-1100-1000-adv-005')
    '''
    pipeline_saverun(['SEED', 'SEED_IV'], architectures.DeepConvNetEncoder, 1000, True, 'DCN-1100-1000-adv-000', lam=0.0)
    pipeline_saverun(['SEED', 'SEED_IV', 'DEAP', 'DREAMER'], architectures.DeepConvNetEncoder, 1000, True, 'DCN-1111-1000-adv-000', lam=0.0)

    pipeline_saverun(['SEED', 'SEED_IV'], architectures.DeepConvNetEncoder, 500, False, 'DCN-1100-500-noa', lam=0.0)
    pipeline_saverun(['SEED', 'SEED_IV', 'DEAP', 'DREAMER'], architectures.DeepConvNetEncoder, 500, False, 'DCN-1111-500-noa', lam=0.0)

    pipeline_saverun(['SEED', 'SEED_IV'], architectures.DeepConvNetEncoder, 100, False, 'DCN-1100-100-noa', lam=0.0)
    pipeline_saverun(['SEED', 'SEED_IV', 'DEAP', 'DREAMER'], architectures.DeepConvNetEncoder, 100, False, 'DCN-1111-100-noa', lam=0.0)

    pipeline_saverun(['SEED', 'SEED_IV'], architectures.DeepConvNetEncoder, 50, False, 'DCN-1100-50-noa', lam=0.0)
    pipeline_saverun(['SEED', 'SEED_IV', 'DEAP', 'DREAMER'], architectures.DeepConvNetEncoder, 50, False, 'DCN-1111-50-noa', lam=0.0)
    