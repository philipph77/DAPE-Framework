import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score
import time
from tqdm import trange, tqdm
import csv
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
        self.use_adversary = False
        self.building_blocks = nn.ModuleList()
        self.building_blocks.extend(encoders)
        self.building_blocks.append(architectures.DenseClassifier(latent_dim, num_classes))

    def forward(self, x_list):
        assert len(self.building_blocks) - 1 == len(x_list)

        z_list = list()
        for datasource_id, x_i in enumerate(x_list):
            z_i = self.building_blocks[datasource_id](x_i)
            z_list.append(z_i)

        # concatenate the outputs
        z = torch.cat(z_list, dim=0)

        # pass z through the classifier
        y_pred = self.building_blocks[-1](z)

        return y_pred

    def add_encoder(self, encoder):
        # ATTENTION: THIS IS NOT SUPPORTED YET!
        if self.is_trained:
            for param in self.parameters():
                param.requires_grad = False
        self.building_blocks.insert(-1,encoder)

    def get_config(self):
        pass

def train(model, train_dataloader, validation_dataloader, run_name, logpath, max_epochs=500, early_stopping_after_epochs=50):
    """Trains a Multi-Source Framework and logs relevant data to file

    Args:
        model ([Framework (PyTorch Model)]): The PyTorch Framework, that you want to train
        train_dataloader ([torch.utils.data.DataLoader]): A DataLoader for the training data
        validation_dataloader ([torch.utils.data.DataLoader]): A DataLoader for the testing data
        run_name ([string]): a unique name, to identify the run later on
        logpath ([string]): the path, where you want to save the logfiles
        max_epochs (int, optional): Maximum number of epochs you want to train. Defaults to 500.
        early_stopping_after_epochs (int, optional): The number of epochs without improvement in validation loss, the training should be stopped afer. Defaults to 50.

    Returns:
        [int]: a status code, 0 - training failed, 1 - training was completed sucessfully
    """    
    #setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    #torch.backends.cudnn.benchmark = True
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    min_loss = np.infty
    early_stopping_wait = 0

    # Prepare Logging
    if not (os.path.isdir(os.path.join(logpath, run_name))):
        os.makedirs(os.path.join(logpath, run_name))

    elif os.path.isfile(os.path.join(logpath, run_name, 'logs.csv')):
        print("LogFile already exists! Rename or remove it, and restart the training")
        return 0
    header = ['Epoch', 'Train-Loss', 'Validation-Loss', 'Validation-Accuracy']
    with open(os.path.join(logpath, run_name, 'logs.csv'), 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(header)

    for epoch in range(1,max_epochs+1):
        # Training
        model.train()
        start_time = time.time()
        train_loss = 0.
        for x_list,y_true in tqdm(train_dataloader):
            #optimizer.zero_grad()
            for param in model.parameters():
                param.grad = None
            x_list = [x_i.to(device) for x_i in x_list]
            y_true = y_true.to(device)
            y_true = torch.flatten(y_true)
            y_pred = model(x_list)
            y_pred.squeeze_()
            loss = criterion(y_pred, y_true)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss = train_loss / len(train_dataloader)
        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0.
            val_acc = 0.
            for x_val_list, y_true in validation_dataloader:
                x_val_list = [x_i.to(device) for x_i in x_val_list]
                y_true = y_true.to(device)
                y_true = torch.flatten(y_true)
                y_val_pred = model(x_val_list)
                y_val_pred.squeeze_()
                val_loss += criterion(y_val_pred,y_true).item()
                val_acc += accuracy_score(y_true.detach().cpu().numpy(), np.argmax(y_val_pred.detach().cpu().numpy(), axis=1))
        val_loss = val_loss / len(validation_dataloader)
        val_acc = val_acc / len(validation_dataloader)
        end_time = time.time()

        # Logging
        print("[%s] Epoch %i: - Train-Loss: %4.2f - Val-Loss: %4.2f - Val-Accuracy: %4.2f - Elapsed Time: %4.2f s"%(run_name, epoch, train_loss, val_loss, val_acc, end_time-start_time))
        with open(os.path.join(logpath, run_name, 'logs.csv'), 'a', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow([str(epoch), str(train_loss), str(val_loss), str(val_acc)])

        # Early Stopping
        if val_loss < min_loss:
            early_stopping_wait=0
            min_loss = val_loss
            best_state = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'optimizer' : optimizer.state_dict(),
                }
        else:
            early_stopping_wait+=1    
            if early_stopping_after_epochs > early_stopping_after_epochs:
                print("Early Stopping")
                break
        
    torch.save(best_state, os.path.join(logpath, run_name, 'best_model.pt'))
    return 1

def test(model, test_dataloader, run_name, logpath):
    """Calculates and logs the achieved accuracy of a trained Framework on a testset

    Args:
        model ([Framework (PyTorch Model)]): The PyTorch Framework, that you want to train
        test_dataloader ([torch.utils.data.DataLoader]): A DataLoader for the training data
        run_name ([string]): a unique name, to identify the run later on
        logpath ([string]): the path, where you want to save the logfiles

    Returns:
        [int]: a status code, 0 - training failed, 1 - training was completed sucessfully
    """    
    if os.path.isfile(os.path.join(logpath, run_name, 'test_logs.csv')):
        print("LogFile already exists! Rename or remove it, and restart the testing")
        return 0
    header = ['Test Accuracy']
    with open(os.path.join(logpath, run_name, 'test_logs.csv'), 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(header)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        acc = 0.
        for x_test_list, y_true in test_dataloader:
            x_test_list = [x_i.to(device) for x_i in x_test_list]
            y_true = y_true.to(device)
            y_true = torch.flatten(y_true)
            y_test_pred = model(x_test_list)
            y_test_pred.squeeze_()
            acc += accuracy_score(y_true.detach().cpu().numpy(), np.argmax(y_test_pred.detach().cpu().numpy(), axis=1))
    acc = acc / len(test_dataloader)
    print("Test-Accuracy: %4.2f"%(acc))

    with open(os.path.join(logpath, run_name, 'test_logs.csv'), 'a', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow([str(acc)])
    
    return 1

def gridsearch(model, training_data, validation_data, batch_sizes=None, num_workers=None):
    import time
    batch_sizes = 2**np.arange(4,11)
    batch_sizes = batch_sizes.tolist()
    num_workers = np.arange(1,9).tolist()
    times = np.empty((len(batch_sizes), len(num_workers)))
    for i,batch_size in enumerate(batch_sizes):
        for j,workers in enumerate(num_workers):
            train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
            validation_dataloader = DataLoader(validation_data, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
            start = time.time()
            train(model, train_dataloader, validation_dataloader, max_epochs=1)
            end = time.time()
            times[i,j] = end-start
            del train_dataloader, validation_dataloader
    print("Each Row is the same batchsize")
    print(times)

if __name__ == '__main__':
    import os
    from torchinfo import summary
    import platform

    if  platform.system() == 'Darwin':
        path = '../../Datasets/private_encs/'
    else:
        path = '../Datasets/private_encs/'
    
    train_datasource_files = [os.path.join(path,'train',f) for f in sorted(os.listdir(os.path.join(path, 'train'))) if f.endswith('.npz') and not('test' in f)]
    validation_datasource_files = [os.path.join(path,'val', f) for f in sorted(os.listdir(os.path.join(path, 'val'))) if f.endswith('.npz') and not('test' in f)]
    test_datasource_files = [os.path.join(path,'test', f) for f in sorted(os.listdir(os.path.join(path, 'test'))) if f.endswith('.npz') and not('test' in f)]

    batch_size = 4*64
    F1, D, F2 = 32, 16, 8
    latent_dim = 20

    encoders = [
        architectures.EEGNetEncoder(channels=32, temporal_filters=F1, spatial_filters=D, pointwise_filters=F2, dropout_propability=0.25, latent_dim=latent_dim, use_constrained_conv=False), #DEAP
        architectures.EEGNetEncoder(channels=14, temporal_filters=F1, spatial_filters=D, pointwise_filters=F2, dropout_propability=0.25, latent_dim=latent_dim, use_constrained_conv=False),  #DREAMER
        architectures.EEGNetEncoder(channels=62, temporal_filters=F1, spatial_filters=D, pointwise_filters=F2, dropout_propability=0.25, latent_dim=latent_dim, use_constrained_conv=False), #SEED
        architectures.EEGNetEncoder(channels=62, temporal_filters=F1, spatial_filters=D, pointwise_filters=F2, dropout_propability=0.25, latent_dim=latent_dim, use_constrained_conv=False) #SEED-IV
    ]
    model = Framework(encoders, 20, 3)

    training_data = datasets.MultiSourceDataset(train_datasource_files)
    validation_data = datasets.MultiSourceDataset(validation_datasource_files)
    batch_sizes = 2**np.arange(4,11)
    batch_sizes = batch_sizes.tolist()
    num_workers = np.arange(1,9).tolist()

    #gridsearch(model, training_data, validation_data, batch_sizes, num_workers)


    #print("----- #Model Parameters: -----")
    #print(helper_funcs.count_parameters(model))
    #print("---------")
    #summary(model, input_size=[(1,62,2*256),(batch_size,1,62,2*256),(batch_size,1,32,2*128),(batch_size,1,14,2*128)])
    #print(model)

    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    validation_dataloader = DataLoader(validation_data, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    if train(model, train_dataloader, validation_dataloader, 'testing_arround', '../logs/', max_epochs=500, early_stopping_after_epochs=1):
        model.is_trained = True

    best_state = torch.load(os.path.join('../logs/', 'testing_arround', 'best_model.pt'))
    model.load_state_dict(best_state['state_dict'])

    test_data = datasets.MultiSourceDataset(test_datasource_files)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    test(model, test_dataloader, 'testing_arround', '../logs')