import sys
sys.path.append("../")
import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import platform
import architectures
from pipeline_funcs import train, train_adversarial, train_singlesource, train_with_mmd_loss, test, test_adversarial, test_singlesource
from torchinfo import summary
from multi_source_framework import Framework
import datasets
import pipeline_helper
import hyperparam_schedulers
from helper_logging import tensorboard_logger, print_logger, csv_logger

def single_source(data_sources, encoder ,latent_dim, train_mode, run_name, version, loss_weight_scheduler, logpath, enc_kwargs=dict(), train_method_kwargs=dict()):
    """runs a single_source DAPE Framework
        For information on the args please refer to pipeline.py
        """   
    if  platform.system() == 'Darwin':
        # MacBook
        path = '../../Datasets/private_encs/'
        BATCHSIZE = 64
        NUM_WORKERS = 1
    else:
        # AWS
        #path = '../../Datasets/private_encs/'
        path = '../../Datasets/private_encs_new/'
        BATCHSIZE = 256
        NUM_WORKERS = 4

    used_hyperparams = {
            'data_sources': pipeline_helper.datasources_to_binary(data_sources),
            'encoder': 'DCN',
            'latent_dim': int(latent_dim),
            'kappa_mode': str(loss_weight_scheduler),
        }

    logging_daemons=[tensorboard_logger(run_name, train_mode, version), print_logger(train_mode), csv_logger(os.path.join(logpath, run_name), train_mode)]
    
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

    model = Framework(encoders, latent_dim, 3, False)

    training_data = datasets.SingleSourceDataset(train_datasource_files[0])
    validation_data = datasets.SingleSourceDataset(validation_datasource_files[0])

    train_dataloader = DataLoader(training_data, batch_size=BATCHSIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    validation_dataloader = DataLoader(validation_data, batch_size=BATCHSIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    train_singlesource(model, train_dataloader, validation_dataloader, run_name, logpath, logging_daemons, max_epochs=300, **train_method_kwargs)

    best_state = torch.load(os.path.join(logpath, run_name, 'best_model.pt'))
    model.load_state_dict(best_state['state_dict'])

    test_data = datasets.SingleSourceDataset(test_datasource_files[0])
    test_dataloader = DataLoader(test_data, batch_size=BATCHSIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    test_singlesource(model, test_dataloader, run_name, logpath, logging_daemons, used_hyperparams)

def pipeline_saverun(data_sources, encoder ,latent_dim, train_mode, run_name, version, loss_weight_scheduler, logpath, enc_kwargs=dict(), train_method_kwargs=dict()):
    try:
        single_source(data_sources, encoder ,latent_dim, train_mode, run_name, version, loss_weight_scheduler, logpath, enc_kwargs, train_method_kwargs)
    except Exception as e:
        pipeline_helper.send_mail_notification('Fehler', run_name, e)
        print(e)

if __name__ == '__main__':
    latent_dim = 50

    for run_id in [0,1,2,3,4]: 
        for data_source in ['SEED', 'SEED_IV', 'DEAP', 'DREAMER']:
            data_source_binary = pipeline_helper.datasources_to_binary([data_source])
            single_source(
                [data_source],
                architectures.DeepConvNetEncoder,
                latent_dim,
                'single-source',
                'DCN-%s-%i-ss-0-vPaper-%i'%(data_source_binary, latent_dim, run_id),
                'vPaper',
                loss_weight_scheduler=hyperparam_schedulers.constant_schedule(value=0.),
                logpath='../logs_vPaper/',
                train_method_kwargs=dict(early_stopping_after_epochs=50)
            )