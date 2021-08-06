import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import platform
import architectures
from pipeline_funcs import train, train_adversarial, train_with_mmd_loss, test, test_adversarial
from torchinfo import summary
from multi_source_framework import Framework
import datasets
import pipeline_helper

def pipeline(data_sources, encoder ,latent_dim, train_mode, run_name, lam=0., logpath='../logs/', **kwargs):
    if  platform.system() == 'Darwin':
        # MacBook
        path = '../../Datasets/private_encs/'
        BATCHSIZE = 64
        NUM_WORKERS = 1
    else:
        # AWS
        path = '../Datasets/private_encs/'
        BATCHSIZE = 256
        NUM_WORKERS = 4

    if train_mode == 'adversarial':
        adversarial = True
    else:
        adversarial = False
    
    train_datasource_files = [os.path.join(path,'train',f) for f in sorted(os.listdir(os.path.join(path, 'train'))) if f.endswith('.npz') and not('test' in f)]
    validation_datasource_files = [os.path.join(path,'val', f) for f in sorted(os.listdir(os.path.join(path, 'val'))) if f.endswith('.npz') and not('test' in f)]
    test_datasource_files = [os.path.join(path,'test', f) for f in sorted(os.listdir(os.path.join(path, 'test'))) if f.endswith('.npz') and not('test' in f)]

    # Select only relevant datasource_files    
    data_sources = sorted(data_sources)
    train_datasource_files = pipeline_helper.filter_datasource_files(train_datasource_files, data_sources)
    validation_datasource_files = pipeline_helper.filter_datasource_files(validation_datasource_files, data_sources)
    test_datasource_files = pipeline_helper.filter_datasource_files(test_datasource_files, data_sources)

    # build the encoder list
    encoders = pipeline_helper.generate_encoder_list(encoder, latent_dim, test_datasource_files, **kwargs)

    model = Framework(encoders, latent_dim, 3, adversarial)

    training_data = datasets.MultiSourceDataset(train_datasource_files)
    validation_data = datasets.MultiSourceDataset(validation_datasource_files)

    train_dataloader = DataLoader(training_data, batch_size=BATCHSIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    validation_dataloader = DataLoader(validation_data, batch_size=BATCHSIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    if adversarial:
        train_adversarial(model, train_dataloader, validation_dataloader, run_name, logpath, lam, max_epochs=300)
    elif train_mode == 'mmd':
        train_with_mmd_loss(model, train_dataloader, validation_dataloader, run_name, logpath, kappa=lam, max_epochs=300)
    else:
        train(model, train_dataloader, validation_dataloader, run_name, logpath, max_epochs=300)

    best_state = torch.load(os.path.join(logpath, run_name, 'best_model.pt'))
    model.load_state_dict(best_state['state_dict'])

    test_data = datasets.MultiSourceDataset(test_datasource_files)
    test_dataloader = DataLoader(test_data, batch_size=BATCHSIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    if adversarial:
        test_adversarial(model, test_dataloader, run_name, logpath)
    else:
        test(model, test_dataloader, run_name, logpath)

def pipeline_saverun(data_sources, encoder ,latent_dim, adversarial, run_name, lam=0., **kwargs):
    try:
        pipeline(data_sources, encoder ,latent_dim, adversarial, run_name, lam, **kwargs)
    except Exception as e:
        pipeline_helper.send_mail_notification('Fehler', run_name)
        print(e)

if __name__ == '__main__':
    num_runs = 5
    latent_dims = [10, 20, 50, 100]
    kappas = [10, 100, 1000, 10000]
    for i in range(num_runs):
        for latent_dim in latent_dims:
            for max_kappa in kappas:
                run_name = "DCN-1111-%i-mmd_scheduled-%2.2f-v5-%i"%(latent_dim, max_kappa, i)
                pipeline_saverun(['SEED', 'SEED_IV', 'DEAP', 'DREAMER'], architectures.DeepConvNetEncoder, latent_dim, 'mmd', run_name, lam=max_kappa, logpath='../logs_v5/')
