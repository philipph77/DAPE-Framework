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
import hyperparam_schedulers
from helper_logging import tensorboard_logger, print_logger, csv_logger

def pipeline(data_sources, encoder ,latent_dim, train_mode, run_name, loss_weight_scheduler, logpath, enc_kwargs=dict(), train_method_kwargs=dict()):
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

    used_hyperparams = {
            'data_sources': pipeline_helper.datasources_to_binary(data_sources),
            'encoder': 'DCN',
            'latent_dim': int(latent_dim),
            'kappa_mode': str(loss_weight_scheduler),
        }

    logging_daemons=[tensorboard_logger(run_name, train_mode), print_logger(train_mode), csv_logger(os.path.join(logpath, run_name), train_mode)]
    
    train_datasource_files = [os.path.join(path,'train',f) for f in sorted(os.listdir(os.path.join(path, 'train'))) if f.endswith('.npz') and not('test' in f)]
    validation_datasource_files = [os.path.join(path,'val', f) for f in sorted(os.listdir(os.path.join(path, 'val'))) if f.endswith('.npz') and not('test' in f)]
    test_datasource_files = [os.path.join(path,'test', f) for f in sorted(os.listdir(os.path.join(path, 'test'))) if f.endswith('.npz') and not('test' in f)]

    # Select only relevant datasource_files    
    data_sources = sorted(data_sources)
    train_datasource_files = pipeline_helper.filter_datasource_files(train_datasource_files, data_sources)
    validation_datasource_files = pipeline_helper.filter_datasource_files(validation_datasource_files, data_sources)
    test_datasource_files = pipeline_helper.filter_datasource_files(test_datasource_files, data_sources)

    # build the encoder list
    encoders = pipeline_helper.generate_encoder_list(encoder, latent_dim, test_datasource_files, **enc_kwargs)

    model = Framework(encoders, latent_dim, 3, adversarial)

    training_data = datasets.MultiSourceDataset(train_datasource_files)
    validation_data = datasets.MultiSourceDataset(validation_datasource_files)

    train_dataloader = DataLoader(training_data, batch_size=BATCHSIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    validation_dataloader = DataLoader(validation_data, batch_size=BATCHSIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    if adversarial:
        train_adversarial(model, train_dataloader, validation_dataloader, run_name, logpath, logging_daemons, lam_scheduler=loss_weight_scheduler, max_epochs=300, **train_method_kwargs)
    elif train_mode == 'mmd':
        train_with_mmd_loss(model, train_dataloader, validation_dataloader, run_name, logpath, logging_daemons, kappa_scheduler=loss_weight_scheduler, max_epochs=300, **train_method_kwargs)
    else:
        train(model, train_dataloader, validation_dataloader, run_name, logpath, logging_daemons, max_epochs=300, **train_method_kwargs)

    best_state = torch.load(os.path.join(logpath, run_name, 'best_model.pt'))
    model.load_state_dict(best_state['state_dict'])

    test_data = datasets.MultiSourceDataset(test_datasource_files)
    test_dataloader = DataLoader(test_data, batch_size=BATCHSIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    if adversarial:
        test_adversarial(model, test_dataloader, run_name, logpath, logging_daemons, used_hyperparams)
    else:
        test(model, test_dataloader, run_name, logpath, logging_daemons, used_hyperparams)

def pipeline_saverun(data_sources, encoder ,latent_dim, train_mode, run_name, loss_weight_scheduler, logpath, enc_kwargs=dict(), train_method_kwargs=dict()):
    try:
        pipeline(data_sources, encoder ,latent_dim, train_mode, run_name, loss_weight_scheduler, logpath, enc_kwargs, train_method_kwargs)
    except Exception as e:
        pipeline_helper.send_mail_notification('Fehler', run_name, e)
        print(e)

if __name__ == '__main__':
    num_runs = 5
    latent_dims = [10, 50, 100, 1000]

    for run_id in range(num_runs):
        for latent_dim in latent_dims:
            pipeline(
            ['SEED', 'SEED_IV', 'DEAP', 'DREAMER'],
            architectures.DeepConvNetEncoder,
            latent_dim,
            'mmd',
            'DCN-1111-%i-mmd-0-v7-%i'%(latent_dim, run_id),
            loss_weight_scheduler=hyperparam_schedulers.constant_schedule(value=0.),
            logpath='../logs_v8/',
            train_method_kwargs=dict(early_stopping_after_epochs=50)
            )

            pipeline_saverun(
            ['SEED', 'SEED_IV', 'DEAP', 'DREAMER'],
            architectures.DeepConvNetEncoder,
            latent_dim,
            'mmd',
            'DCN-1111-%i-mmd-clc-v7-%i'%(latent_dim, run_id),
            loss_weight_scheduler=hyperparam_schedulers.constant_linear_constant_schedule(start_epoch=5, start_value=0, step_value=0.25, stop_epoch=70),
            logpath='../logs_v8/',
            train_method_kwargs=dict(early_stopping_after_epochs=50)
            )

    