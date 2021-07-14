from sklearn.utils import shuffle


def test_single_source():
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
    latent_dim = 2000

    encoders = [
        architectures.EEGNetEncoder(channels=32, temporal_filters=F1, spatial_filters=D, pointwise_filters=F2, dropout_propability=0.25, latent_dim=latent_dim, use_constrained_conv=True), #DEAP
        architectures.EEGNetEncoder(channels=14, temporal_filters=F1, spatial_filters=D, pointwise_filters=F2, dropout_propability=0.25, latent_dim=latent_dim, use_constrained_conv=True),  #DREAMER
        architectures.EEGNetEncoder(channels=62, temporal_filters=F1, spatial_filters=D, pointwise_filters=F2, dropout_propability=0.25, latent_dim=latent_dim, use_constrained_conv=True), #SEED
        architectures.EEGNetEncoder(channels=62, temporal_filters=F1, spatial_filters=D, pointwise_filters=F2, dropout_propability=0.25, latent_dim=latent_dim, use_constrained_conv=True) #SEED-IV
    ]
    encoders = [architectures.EEGNetEncoder(channels=62, temporal_filters=F1, spatial_filters=D, pointwise_filters=F2, dropout_propability=0.25, latent_dim=latent_dim, use_constrained_conv=True)] #SEED
    encoders = [architectures.DeepConvNetEncoder(channels=62, latent_dim=latent_dim)]

    model = Framework(encoders, latent_dim, 3, use_adversary=True)
    print(train_datasource_files)
    train_datasource_files = [f for f in train_datasource_files if 'SEED.' in f]
    print(train_datasource_files)
    validation_datasource_files = [f for f in validation_datasource_files if 'SEED.' in f]

    training_data = datasets.MultiSourceDataset(train_datasource_files)
    validation_data = datasets.MultiSourceDataset(validation_datasource_files)

    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    validation_dataloader = DataLoader(validation_data, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    
    #train(model, train_dataloader, validation_dataloader, 'only-seed-DCN-2000', '../logs/', max_epochs=30)
    train_adversarial(model, train_dataloader, validation_dataloader, 'only-seed-DCN-2000-adv', '../logs/',  0.05, max_epochs=30)

if __name__ == '__main__':
    #test_single_source()
    myl = [1,2,3,4,5,6]
    print([myl[f] for f in range(len(myl)) if not(f==len(myl)-2)])