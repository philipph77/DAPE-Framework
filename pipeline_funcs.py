import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from sklearn.metrics import accuracy_score
import time
from tqdm import trange, tqdm
import csv
from pipeline_helper import MMD_loss


def train(model, train_dataloader, validation_dataloader, run_name, logpath, max_epochs=500, early_stopping_after_epochs=20):
    """Trains a Multi-Source Framework and logs relevant data to file

    Args:
        model ([Framework (PyTorch Model)]): The PyTorch Framework, that you want to train
        train_dataloader ([torch.utils.data.DataLoader]): A DataLoader for the training data
        validation_dataloader ([torch.utils.data.DataLoader]): A DataLoader for the testing data
        run_name ([string]): a unique name, to identify the run later on
        logpath ([string]): the path, where you want to save the logfiles
        max_epochs (int, optional): Maximum number of epochs you want to train. Defaults to 500.
        early_stopping_after_epochs (int, optional): The number of epochs without improvement in validation loss, the training should be stopped afer. Defaults to 20.

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
        for x_list,y_true, _ in tqdm(train_dataloader):
            #optimizer.zero_grad()
            for param in model.parameters():
                param.grad = None
            x_list = [x_i.to(device) for x_i in x_list]
            y_true = y_true.to(device)
            y_true = torch.flatten(torch.transpose(y_true,0,1))
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
            for x_val_list, y_true, _ in validation_dataloader:
                x_val_list = [x_i.to(device) for x_i in x_val_list]
                y_true = y_true.to(device)
                y_true = torch.flatten(torch.transpose(y_true,0,1))
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
            if early_stopping_wait > early_stopping_after_epochs:
                print("Early Stopping")
                break
        
    torch.save(best_state, os.path.join(logpath, run_name, 'best_model.pt'))
    return 1

def train_adversarial(model, train_dataloader, validation_dataloader, run_name, logpath, lam=0.05, max_epochs=500, early_stopping_after_epochs=20):
    """Trains a Multi-Source Framework and logs relevant data to file

    Args:
        model ([Framework (PyTorch Model)]): The PyTorch Framework, that you want to train
        train_dataloader ([torch.utils.data.DataLoader]): A DataLoader for the training data
        validation_dataloader ([torch.utils.data.DataLoader]): A DataLoader for the testing data
        run_name ([string]): a unique name, to identify the run later on
        logpath ([string]): the path, where you want to save the logfiles
        lam ([float]): loss weight for the adversary
        max_epochs (int, optional): Maximum number of epochs you want to train. Defaults to 500.
        early_stopping_after_epochs (int, optional): The number of epochs without improvement in validation loss, the training should be stopped afer. Defaults to 20.

    Returns:
        [int]: a status code, 0 - training failed, 1 - training was completed sucessfully
    """    
    #setup
    torch.autograd.set_detect_anomaly(True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    #torch.backends.cudnn.benchmark = True
    # TODO: make the following optimizer for all except the adversary
    enc_params = list()
    for i in range(len(model.encoders)):
        enc_params += list(model.encoders[i].parameters())
    cla_params = model.emotion_classifier[0].parameters()
    adv_params = model.domain_classifier[0].parameters()
    cla_optimizer = optim.Adam(list(enc_params) + list(cla_params), lr=1e-3, weight_decay=1e-4)
    adv_optimizer = optim.Adam(adv_params, lr=1e-3, weight_decay=1e-4) 
    cla_criterion = nn.CrossEntropyLoss()
    adv_criterion = nn.CrossEntropyLoss()
    min_total_loss = np.infty
    early_stopping_wait = 0

    # Prepare Logging
    if not (os.path.isdir(os.path.join(logpath, run_name))):
        os.makedirs(os.path.join(logpath, run_name))

    elif os.path.isfile(os.path.join(logpath, run_name, 'logs.csv')):
        print("LogFile already exists! Rename or remove it, and restart the training")
        return 0
    header = ['Epoch', 'Total-Train-Loss', 'CLA-Train-Loss', 'ADV-Train-Loss', 'Total-Validation-Loss', 'CLA-Validation-Loss', 'ADV-Validation-Loss', 'CLA-Validation-Accuracy', 'ADV-Validation-Accuracy']
    with open(os.path.join(logpath, run_name, 'logs.csv'), 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(header)

    for epoch in range(1,max_epochs+1):
        # Training
        model.train()
        start_time = time.time()
        cla_train_loss = 0.
        adv_train_loss = 0.
        total_train_loss = 0.
        for x_list, y_true, d_true in tqdm(train_dataloader):
            #optimizer.zero_grad()
            for param in model.parameters():
                param.grad = None
            x_list = [x_i.to(device) for x_i in x_list]
            y_true = y_true.to(device)
            d_true = d_true.to(device)
            y_true = torch.flatten(torch.transpose(y_true,0,1))
            d_true = torch.flatten(torch.transpose(d_true,0,1))
            y_pred, d_pred = model(x_list)
            y_pred.squeeze_()
            d_pred.squeeze_(1)
            adv_loss = adv_criterion(d_pred, d_true)
            adv_loss.backward()
            adv_optimizer.step()
            cla_optimizer.zero_grad()
            y_pred, d_pred = model(x_list)
            y_pred.squeeze_()
            d_pred.squeeze_(1)
            cla_loss = cla_criterion(y_pred, y_true)
            adv_loss = adv_criterion(d_pred, d_true)
            total_loss = cla_loss - lam*adv_loss
            total_loss.backward()
            cla_optimizer.step()
            cla_train_loss += cla_loss.item()
            adv_train_loss += adv_loss.item()
            total_train_loss += total_loss.item()
        cla_train_loss = cla_train_loss / len(train_dataloader)
        adv_train_loss = adv_train_loss / len(train_dataloader)
        total_train_loss = total_train_loss / len(train_dataloader)
        # Validation
        model.eval()
        with torch.no_grad():
            cla_val_loss = 0.
            adv_val_loss = 0.
            total_val_loss = 0.
            cla_val_acc = 0.
            adv_val_acc = 0.
            for x_val_list, y_true, d_true in validation_dataloader:
                x_val_list = [x_i.to(device) for x_i in x_val_list]
                y_true = y_true.to(device)
                d_true = d_true.to(device)
                y_true = torch.flatten(torch.transpose(y_true,0,1))
                d_true = torch.flatten(torch.transpose(d_true,0,1))
                y_val_pred, d_val_pred = model(x_val_list)
                y_val_pred.squeeze_()
                d_val_pred.squeeze_(1)
                cla_val_loss += cla_criterion(y_val_pred,y_true).item()
                adv_val_loss += adv_criterion(d_val_pred,d_true).item()
                cla_val_acc += accuracy_score(y_true.detach().cpu().numpy(), np.argmax(y_val_pred.detach().cpu().numpy(), axis=1))
                adv_val_acc += accuracy_score(d_true.detach().cpu().numpy(), np.argmax(d_val_pred.detach().cpu().numpy(), axis=1))
        cla_val_loss = cla_val_loss / len(validation_dataloader)
        adv_val_loss = adv_val_loss / len(validation_dataloader)
        total_val_loss = cla_val_loss - lam * adv_val_loss

        cla_val_acc = cla_val_acc / len(validation_dataloader)
        adv_val_acc = adv_val_acc / len(validation_dataloader)
        
        end_time = time.time()

        # Logging
        print("[%s] Epoch %i: - Total-Train-Loss: %4.2f -  CLA-Train-Loss: %4.2f - ADV-Train-Loss: %4.2f - Total-Val-Loss: %4.2f - CLA-Val-Loss: %4.2f - ADV-Val-Loss: %4.2f - CLA-Val-Accuracy: %4.2f - ADV-Val-Accuracy: %4.2f - Elapsed Time: %4.2f s" \
        %(run_name, epoch, total_train_loss, cla_train_loss, adv_train_loss, total_val_loss, cla_val_loss, adv_val_loss, cla_val_acc, adv_val_acc, end_time-start_time))
        with open(os.path.join(logpath, run_name, 'logs.csv'), 'a', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow([str(epoch), str(total_train_loss), str(cla_train_loss), str(adv_train_loss), str(total_val_loss), str(cla_val_loss), str(adv_val_loss), str(cla_val_acc), str(adv_val_acc)])

        # Early Stopping
        if cla_val_loss < min_total_loss:
            early_stopping_wait=0
            min_total_loss = cla_val_loss
            best_state = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'total_train_loss': total_train_loss,
                    'cla_train_loss': cla_train_loss,
                    'adv_train_loss': adv_train_loss,
                    'total_val_loss': total_val_loss,
                    'cla_val_loss': cla_val_loss,
                    'adv_val_loss': adv_val_loss,
                    'cla_val_acc': cla_val_acc,
                    'adv_val_acc': adv_val_acc,
                    'cla_optimizer' : cla_optimizer.state_dict(),
                    'adv_optimizer' : adv_optimizer.state_dict(),
                    'lam': lam,
                }
        else:
            early_stopping_wait+=1    
            if early_stopping_wait > early_stopping_after_epochs:
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
        for x_test_list, y_true, _ in test_dataloader:
            x_test_list = [x_i.to(device) for x_i in x_test_list]
            y_true = y_true.to(device)
            y_true = torch.flatten(torch.transpose(y_true,0,1))
            y_test_pred = model(x_test_list)
            y_test_pred.squeeze_()
            acc += accuracy_score(y_true.detach().cpu().numpy(), np.argmax(y_test_pred.detach().cpu().numpy(), axis=1))
    acc = acc / len(test_dataloader)
    print("Test-Accuracy: %4.2f"%(acc))

    with open(os.path.join(logpath, run_name, 'test_logs.csv'), 'a', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow([str(acc)])
    
    return 1

def test_adversarial(model, test_dataloader, run_name, logpath):
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
    header = ['CLA Test Accuracy', 'ADV Test Accuracy']
    with open(os.path.join(logpath, run_name, 'test_logs.csv'), 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(header)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        cla_test_acc = 0.
        adv_test_acc = 0.
        for x_test_list, y_true, d_true in test_dataloader:
            x_test_list = [x_i.to(device) for x_i in x_test_list]
            y_true = y_true.to(device)
            d_true = d_true.to(device)
            y_true = torch.flatten(torch.transpose(y_true,0,1))
            d_true = torch.flatten(torch.transpose(d_true,0,1))
            y_test_pred, d_test_pred = model(x_test_list)
            y_test_pred.squeeze_()
            d_test_pred.squeeze_(1)
            cla_test_acc += accuracy_score(y_true.detach().cpu().numpy(), np.argmax(y_test_pred.detach().cpu().numpy(), axis=1))
            adv_test_acc += accuracy_score(d_true.detach().cpu().numpy(), np.argmax(d_test_pred.detach().cpu().numpy(), axis=1))
    cla_test_acc = cla_test_acc / len(test_dataloader)
    adv_test_acc = adv_test_acc / len(test_dataloader)
    print("CLA-Test-Accuracy: %4.2f - ADV-Test_Accuracy: %4.2f"%(cla_test_acc, adv_test_acc))

    with open(os.path.join(logpath, run_name, 'test_logs.csv'), 'a', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow([str(cla_test_acc), str(adv_test_acc)])
    
    return 1

def pretrain_encoders(model, train_dataloader, validation_dataloader, run_name, logpath, max_epochs=25):
    #setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizers = list()
    criteria = list()
    min_loss = list()
    for encoder_id, encoder in enumerate(model.encoders):
        optimizers.append(optim.Adam(list(encoder.parameters()) + list(model.individual_classifiers[encoder_id].parameters()), lr=1e-3, weight_decay=1e-4))
        criteria.append(nn.CrossEntropyLoss())
        min_loss.append(np.infty)

    # Prepare Logging
    if not (os.path.isdir(os.path.join(logpath, run_name))):
        os.makedirs(os.path.join(logpath, run_name))

    elif os.path.isfile(os.path.join(logpath, run_name, 'pretrain_logs.csv')):
        print("LogFile already exists! Rename or remove it, and restart the training")
        return 0
    header = ['Epoch', 'Data-Source-ID', 'Train-Loss', 'Validation-Loss', 'Validation-Accuracy']
    with open(os.path.join(logpath, run_name, 'pretrain_logs.csv'), 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(header)

    for epoch in range(1,max_epochs+1):
        # Training
        model.train()
        train_loss = list()
        loss = list()
        best_state = list()
        for _ in range(len(model.encoders)):
            train_loss.append(0.)
            loss.append(None)
            best_state.append(None)
        for x_list,y_true, _ in tqdm(train_dataloader):
            #optimizer.zero_grad()
            for param in model.parameters():
                param.grad = None
            x_list = [x_i.to(device) for x_i in x_list]
            y_true_list = list()
            for i in range(y_true.shape[1]):
                y_true_list.append(y_true[:,i])
            y_true_list = [y_true_i.to(device) for y_true_i in y_true_list]
            _, y_pred_individual = model(x_list, individual_outputs=True)
            for i in range(len(y_pred_individual)):
                y_pred_individual[i].squeeze_()
                loss[i] = criteria[i](y_pred_individual[i], y_true_list[i])
                loss[i].backward()
                optimizers[i].step()
                train_loss[i] += loss[i].item()
        train_loss = [l/len(train_dataloader) for l in train_loss]
        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = list()
            val_acc = list()
            for _ in range(len(model.encoders)):
                val_loss.append(0.)
                val_acc.append(0.)
            for x_val_list, y_true, _ in validation_dataloader:
                x_val_list = [x_i.to(device) for x_i in x_val_list]
                y_true_list = list()
                for i in range(y_true.shape[1]):
                    y_true_list.append(y_true[:,i])
                y_true_list = [y_true_i.to(device) for y_true_i in y_true_list]
                _, y_val_pred = model(x_val_list, individual_outputs=True)
                for i in range(len(y_val_pred)):
                    y_val_pred[i].squeeze_()
                    val_loss[i] += criteria[i](y_val_pred[i],y_true_list[i]).item()
                    val_acc[i] += accuracy_score(y_true_list[i].detach().cpu().numpy(), np.argmax(y_val_pred[i].detach().cpu().numpy(), axis=1))
        val_loss = [vl / len(validation_dataloader) for vl in val_loss]
        val_acc = [va / len(validation_dataloader) for va in val_acc]

        # Logging
        for i, encoder in enumerate(model.encoders):
            print("[%s-%i] Pretrain-Epoch %i: - Train-Loss: %4.2f - Val-Loss: %4.2f - Val-Accuracy: %4.2f"%(run_name, i, epoch, train_loss[i], val_loss[i], val_acc[i]))
            with open(os.path.join(logpath, run_name, 'logs.csv'), 'a', encoding='UTF8') as f:
                writer = csv.writer(f)
                writer.writerow([str(epoch), str(i), str(train_loss[i]), str(val_loss[i]), str(val_acc[i])])
            if val_loss[i] < min_loss[i]:
                min_loss[i] = val_loss[i]
                best_state[i] = {
                        'data_source_id': i,
                        'epoch': epoch,
                        'encoder_state_dict': encoder.state_dict(),
                        'classifier_state_dict': model.individual_classifiers[i].state_dict(),
                        'train_loss': train_loss[i],
                        'val_loss': val_loss[i],
                        'val_acc': val_acc[i],
                        'optimizer' : optimizers[i].state_dict(),
                    }
        
    for i in range(len(model.encoders)):
        torch.save(best_state[i], os.path.join(logpath, run_name, 'pretrained_encoder_'+str(i)+'.pt'))
    
    return 1

def train_with_mmd_loss(model, train_dataloader, validation_dataloader, run_name, logpath, kappa = 0.05, max_epochs=500, early_stopping_after_epochs=20):
    """Trains a Multi-Source Framework and logs relevant data to file
        A Maximum-Mean-Discrepancy-Loss Term is added to the CE-Loss, in order to align the distributions from the encoder
    Args:
        model ([Framework (PyTorch Model)]): The PyTorch Framework, that you want to train
        train_dataloader ([torch.utils.data.DataLoader]): A DataLoader for the training data
        validation_dataloader ([torch.utils.data.DataLoader]): A DataLoader for the testing data
        run_name ([string]): a unique name, to identify the run later on
        logpath ([string]): the path, where you want to save the logfiles
        max_epochs (int, optional): Maximum number of epochs you want to train. Defaults to 500.
        early_stopping_after_epochs (int, optional): The number of epochs without improvement in validation loss, the training should be stopped afer. Defaults to 20.

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
    header = ['Epoch', 'Total-Train-Loss', 'CLA-Train-Loss', 'MMD-Train-Loss', 'Total-Validation-Loss', 'CLA-Validation-Loss', 'MMD-Validation-Loss', 'Validation-Accuracy']
    with open(os.path.join(logpath, run_name, 'logs.csv'), 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(header)

    for epoch in range(1,max_epochs+1):
        # Training
        model.train()
        start_time = time.time()
        total_train_loss = 0.
        total_mmd_loss = 0.
        total_ce_loss = 0.
        for x_list,y_true, _ in tqdm(train_dataloader):
            #optimizer.zero_grad()
            for param in model.parameters():
                param.grad = None
            x_list = [x_i.to(device) for x_i in x_list]
            y_true = y_true.to(device)
            y_true = torch.flatten(torch.transpose(y_true,0,1))
            y_pred, z_list = model(x_list, output_latent_representation=True)
            y_pred.squeeze_()
            ce_loss = criterion(y_pred, y_true)
            mmd_loss = MMD_loss(z_list, 'rbf', 4)
            total_loss = ce_loss + kappa * mmd_loss
            total_loss.backward()
            optimizer.step()
            total_ce_loss += ce_loss.item()
            total_mmd_loss += mmd_loss.item()
            total_train_loss += total_loss.item()
        total_ce_loss = total_ce_loss / len(train_dataloader)
        total_mmd_loss = total_mmd_loss / len(train_dataloader)
        total_train_loss = total_train_loss / len(train_dataloader)
        # Validation
        model.eval()
        with torch.no_grad():
            total_val_loss = 0.
            total_val_mmd_loss = 0.
            total_val_ce_loss = 0.
            val_acc = 0.
            for x_val_list, y_true, _ in validation_dataloader:
                x_val_list = [x_i.to(device) for x_i in x_val_list]
                y_true = y_true.to(device)
                y_true = torch.flatten(torch.transpose(y_true,0,1))
                y_val_pred, z_val_list = model(x_val_list, output_latent_representation=True)
                y_val_pred.squeeze_()
                total_val_ce_loss += criterion(y_val_pred,y_true).item()
                total_val_mmd_loss += MMD_loss(z_val_list, 'rbf', 4).item()
                val_acc += accuracy_score(y_true.detach().cpu().numpy(), np.argmax(y_val_pred.detach().cpu().numpy(), axis=1))
        total_val_ce_loss = total_val_ce_loss / len(validation_dataloader)
        total_val_mmd_loss = total_val_mmd_loss / len(validation_dataloader)
        total_val_loss += total_val_ce_loss + kappa * total_val_mmd_loss
        val_acc = val_acc / len(validation_dataloader)
        end_time = time.time()

        # Logging
        header = ['Epoch', 'Total-Train-Loss', 'CLA-Train-Loss', 'MMD-Train-Loss', 'Total-Validation-Loss', 'CLA-Validation-Loss', 'MMD-Validation-Loss', 'Validation-Accuracy']
        print("[%s] Epoch %i: - Total-Train-Loss: %4.2f - CLA-Train-Loss: %4.2f - MMD-Train-Loss: %4.2f - Total-Val-Loss: %4.2f - CLA-Val-Loss: %4.2f - MMD-Validation-Loss: %4.2f - Val-Accuracy: %4.2f - Elapsed Time: %4.2f s"%(
            run_name, epoch, total_train_loss, total_ce_loss, total_mmd_loss, total_val_loss, total_val_ce_loss, total_val_mmd_loss, val_acc, end_time-start_time))
        with open(os.path.join(logpath, run_name, 'logs.csv'), 'a', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow([str(epoch), str(total_train_loss), str(total_ce_loss), str(total_mmd_loss), str(total_val_loss), str(total_val_ce_loss), str(total_val_mmd_loss), str(val_acc)])

        # Early Stopping
        if total_val_loss < min_loss:
            early_stopping_wait=0
            min_loss = total_val_loss
            best_state = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'total_train_loss': total_train_loss,
                    'cla_train_loss': total_ce_loss,
                    'mmd_train_loss': total_mmd_loss,
                    'total_val_loss': total_val_loss,
                    'cla_val_loss': total_val_ce_loss,
                    'mmd_val_loss': total_val_mmd_loss,
                    'val_acc': val_acc,
                    'optimizer' : optimizer.state_dict(),
                    'kappa': kappa,
                }
        else:
            early_stopping_wait+=1    
            if early_stopping_wait > early_stopping_after_epochs:
                print("Early Stopping")
                break
        
    torch.save(best_state, os.path.join(logpath, run_name, 'best_model.pt'))
    return 1
