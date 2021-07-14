if __name__ == '__main__':
    import os
    import platform
    import architectures
    import datasets
    from torch.utils.data import DataLoader
    from multi_source_framework import Framework
    from pipeline_funcs import train, train_adversarial

    if  platform.system() == 'Darwin':
        path = '../../Datasets/private_encs/'
    else:
        path = '../Datasets/private_encs/'
    
    train_datasource_files = [os.path.join(path,'train',f) for f in sorted(os.listdir(os.path.join(path, 'train'))) if f.endswith('.npz') and not('test' in f)]
    validation_datasource_files = [os.path.join(path,'val', f) for f in sorted(os.listdir(os.path.join(path, 'val'))) if f.endswith('.npz') and not('test' in f)]

    F1, D, F2 = 32, 16, 8
    latent_dim = 1000

    encoders = [
        architectures.EEGNetEncoder(channels=32, temporal_filters=F1, spatial_filters=D, pointwise_filters=F2, dropout_propability=0.25, latent_dim=latent_dim, use_constrained_conv=False), #DEAP
        architectures.EEGNetEncoder(channels=14, temporal_filters=F1, spatial_filters=D, pointwise_filters=F2, dropout_propability=0.25, latent_dim=latent_dim, use_constrained_conv=False),  #DREAMER
        architectures.EEGNetEncoder(channels=62, temporal_filters=F1, spatial_filters=D, pointwise_filters=F2, dropout_propability=0.25, latent_dim=latent_dim, use_constrained_conv=False), #SEED
        architectures.EEGNetEncoder(channels=62, temporal_filters=F1, spatial_filters=D, pointwise_filters=F2, dropout_propability=0.25, latent_dim=latent_dim, use_constrained_conv=False) #SEED-IV
    ]
    encoders = [architectures.EEGNetEncoder(channels=62, temporal_filters=F1, spatial_filters=D, pointwise_filters=F2, dropout_propability=0.25, latent_dim=latent_dim, use_constrained_conv=False)] #SEED

    model = Framework(encoders, latent_dim, 3, use_adversary=False)

    train_datasource_files = [f for f in train_datasource_files if 'SEED.' in f]
    validation_datasource_files = [f for f in validation_datasource_files if 'SEED.' in f]

    training_data = datasets.MultiSourceDataset(train_datasource_files)
    validation_data = datasets.MultiSourceDataset(validation_datasource_files)

    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    validation_dataloader = DataLoader(validation_data, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    train(model, train_dataloader, validation_dataloader, 'only-seed', '../logs/', max_epochs=500)