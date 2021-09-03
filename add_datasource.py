import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import csv
import time
from tqdm import tqdm, trange
from pipeline_helper import MMD_loss
from sklearn.metrics import accuracy_score
from sklearn.utils import resample

def train_domain_aligned_autoencoder(autoencoder, train_epochs, train_dataloader, validation_dataloader, z_train_list, z_val_list, run_name, logpath, kappa_scheduler, early_stopping_after_epochs=100, scheduler_kwargs=dict()):
    """CAUTION: THIS METHOD IS NOT TESTED YET!
    This method trains an autoencoder on the given data

    Args:
        autoencoder (PyTorch Model): the autoencoder that shall be trained
        train_epochs (int): number of epochs the model should be trained
        train_dataloader (PyTorch Dataloader): dataloader for the training data
        validation_dataloader (PyTorch Dataloader): datalaoder for the validation data
        z_train_list ([list of numpy arrays]): latent representations to wich the representation in the bottleneck of the autoencoder should be aligned
        z_val_list ([list of numpy arrays]): latent representations to wich the representation in the bottleneck of the autoencoder should be aligned
        run_name (string): identifier
        logpath (string): path, where the logfiles shall be stored
        kappa_scheduler ([hyperparam_scheduler object]): scheduler for the MMD-loss weight kappa
        early_stopping_after_epochs (int, optional): number of epochs withouth validation loss improvement, after wich training shall be stopped. Defaults to 100.
        scheduler_kwargs (dict, optional): kwargs that shall be passed to the kappa_scheduler. Defaults to dict().

    Returns:
        [pytorch model]: trained autoencoder
    """    
    # train autoencoder with MMD Alignment
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    autoencoder = autoencoder.to(device)
    
    #torch.backends.cudnn.benchmark = True
    optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3, weight_decay=1e-4)
    ae_criterion = nn.MSELoss()
    min_loss = np.infty
    early_stopping_wait = 0

    # Prepare Logging
    if not (os.path.isdir(os.path.join(logpath, run_name))):
        os.makedirs(os.path.join(logpath, run_name))

    elif os.path.isfile(os.path.join(logpath, run_name, 'autoencoder_logs.csv')):
        print("LogFile already exists! Rename or remove it, and restart the training")
        return 0
    header = ['Epoch', 'Total-Train-Loss', 'AE-Train-Loss', 'MMD-Train-Loss', 'Total-Validation-Loss', 'AE-Validation-Loss', 'MMD-Validation-Loss', 'Validation-Accuracy']
    with open(os.path.join(logpath, run_name, 'logs.csv'), 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(header)

    for epoch in range(1,train_epochs+1):
        # Training
        autoencoder.train()
        start_time = time.time()
        total_train_loss = 0.
        total_mmd_loss = 0.
        total_ae_loss = 0.
        kappa = kappa_scheduler(epoch, **scheduler_kwargs)
        for x_true, _ in tqdm(train_dataloader):
            #optimizer.zero_grad()
            for param in autoencoder.parameters():
                param.grad = None
            x_true = x_true.to(device)
            x_pred, z_pred = autoencoder(x_true)
            ae_loss = ae_criterion(x_pred, x_true)
            z_train_list_resampled = [resample(z, replace=False, n_samples=z_pred.shape[0], random_state=7) for z in z_train_list]
            mmd_loss = MMD_loss(z_train_list_resampled, 'rbf', 1, z_pred)
            total_loss = ae_loss + kappa * mmd_loss
            total_loss.backward()
            optimizer.step()
            total_ae_loss += ae_loss.item()
            total_mmd_loss += mmd_loss.item()
            total_train_loss += total_loss.item()
        total_ae_loss = total_ae_loss / len(train_dataloader)
        total_mmd_loss = total_mmd_loss / len(train_dataloader)
        total_train_loss = total_train_loss / len(train_dataloader)
        # Validation
        autoencoder.eval()
        with torch.no_grad():
            total_val_loss = 0.
            total_val_mmd_loss = 0.
            total_val_ae_loss = 0.
            val_acc = 0.
            for x_val_true, _ in validation_dataloader:
                x_val_true = x_val_true.to(device)
                x_val_pred, z_val_pred = autoencoder(x_val_true)
                total_val_ae_loss += ae_criterion(x_val_pred,x_val_true).item()
                z_val_list_resampled = [resample(z, replace=False, n_samples=z_val_pred.shape[0], random_state=7) for z in z_val_list]
                total_val_mmd_loss += MMD_loss(z_val_list_resampled, 'rbf', 1, z_val_pred).item()
        total_val_ae_loss = total_val_ae_loss / len(validation_dataloader)
        total_val_mmd_loss = total_val_mmd_loss / len(validation_dataloader)
        total_val_loss += total_val_ae_loss + kappa * total_val_mmd_loss
        end_time = time.time()

        # Logging
        print("[%s] Epoch %i: - kappa: %4.2f - Total-Train-Loss: %4.2f - AE-Train-Loss: %4.2f - MMD-Train-Loss: %4.2f - Total-Val-Loss: %4.2f - AE-Val-Loss: %4.2f - MMD-Validation-Loss: %4.2f - Elapsed Time: %4.2f s"%(
            run_name,  epoch, kappa, total_train_loss, total_ae_loss, total_mmd_loss, total_val_loss, total_val_ae_loss, total_val_mmd_loss, end_time-start_time))
        with open(os.path.join(logpath, run_name, 'autoencoder_logs.csv'), 'a', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow([str(epoch), str(total_train_loss), str(total_ae_loss), str(total_mmd_loss), str(total_val_loss), str(total_val_ae_loss), str(total_val_mmd_loss), str(val_acc)])

        # Early Stopping
        if total_val_loss < min_loss:
            early_stopping_wait=0
            min_loss = total_val_loss
            best_state = {
                    'epoch': epoch,
                    'state_dict': autoencoder.state_dict(),
                    'total_train_loss': total_train_loss,
                    'cla_train_loss': total_ae_loss,
                    'mmd_train_loss': total_mmd_loss,
                    'total_val_loss': total_val_loss,
                    'cla_val_loss': total_val_ae_loss,
                    'mmd_val_loss': total_val_mmd_loss,
                    'optimizer' : optimizer.state_dict(),
                    'kappa': kappa,
                }
        else:
            early_stopping_wait+=1    
            if early_stopping_wait > early_stopping_after_epochs:
                print("Early Stopping")
                break
        
    torch.save(best_state, os.path.join(logpath, run_name, 'best_autoencoder.pt'))

    return autoencoder


def test_new_datasource(encoder, classifier, test_dataloader, logpath, run_name):
    if os.path.isfile(os.path.join(logpath, run_name, 'test_logs.csv')):
        print("LogFile already exists! Rename or remove it, and restart the testing")
        return 0
    header = ['Test Accuracy']
    with open(os.path.join(logpath, run_name, 'test_logs.csv'), 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(header)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(device)
    classifier = classifier.to(device)
    encoder.eval()
    classifier.eval()
    with torch.no_grad():
        acc = 0.
        for x_test, y_true, _ in test_dataloader:
            x_test = x_test.to(device)
            y_true = y_true.to(device)
            y_true = torch.flatten(torch.transpose(y_true,0,1))
            y_test_pred = classifier(encoder(x_test))
            y_test_pred.squeeze_()
            acc += accuracy_score(y_true.detach().cpu().numpy(), np.argmax(y_test_pred.detach().cpu().numpy(), axis=1))
    acc = acc / len(test_dataloader)
    print("Test-Accuracy: %4.2f"%(acc))

    with open(os.path.join(logpath, run_name, 'test_logs.csv'), 'a', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow([str(acc)])
    
    return 1

if __name__ == '__main__':
    import architectures
    from autoencoder import autoencoder
    from datasets import SingleSourceDataset, MultiSourceDataset
    from torch.utils.data import DataLoader
    from hyperparam_schedulers import constant_schedule


    BATCHSIZE = 64
    LATENTDIM = 100

    encoder = architectures.DeepConvNetEncoder(32, LATENTDIM)
    decoder = architectures.DeepConvNetDecoder(LATENTDIM, 256, 32)
    ae = autoencoder(encoder, decoder)

    trainset = SingleSourceDataset('../../Datasets/private_encs/train/class_stratified_DEAP.npz', supress_output=True)
    valset = SingleSourceDataset('../../Datasets/private_encs/val/class_stratified_DEAP.npz', supress_output=True)


    train_dataloader = DataLoader(trainset, batch_size=BATCHSIZE, shuffle=True, num_workers=1, pin_memory=True)
    validation_dataloader = DataLoader(valset, batch_size=BATCHSIZE, shuffle=False, num_workers=1, pin_memory=True)

    z_train_list = list()
    z_val_list = list()
    for i in range(3):
        z_train_list.append(torch.randn(6138, 1, LATENTDIM))
        z_val_list.append(torch.randn(6138, 1, LATENTDIM))

    train_domain_aligned_autoencoder(ae, 200, train_dataloader, validation_dataloader, z_train_list, z_val_list, 'testing-autoencoder-training', '../delete/', constant_schedule, early_stopping_after_epochs=100, scheduler_kwargs=dict(value=0))

    