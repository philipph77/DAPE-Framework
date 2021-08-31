import os
import csv
import torch
from torch.utils.tensorboard import SummaryWriter

class tensorboard_logger():
    def __init__(self, run_name, train_mode, version, write_embeddings=False):
        self.run_name = run_name
        self.writer = SummaryWriter('../logs_'+version+'/tensorboard_logs/'+self.run_name+'/')
        self.train_mode = train_mode
        self.write_embeddings = write_embeddings
        #print("Hi - I am going to handle your Tensorboard Logs")

    def write_state(self,state):
        ## Losses
        self.writer.add_scalar("01-Train-Loss/Total-Train-Loss", state['scalars']['total-train-loss'], state['epoch'])
        self.writer.add_scalar("02-Validation-Loss/Total-Validation-Loss", state['scalars']['total-val-loss'], state['epoch'])

        if self.train_mode == 'mmd':
            self.writer.add_scalar("01-Train-Loss/MMD-Train-Loss", state['scalars']['mmd-train-loss'], state['epoch'])
            self.writer.add_scalar("01-Train-Loss/CE-Train-Loss", state['scalars']['ce-train-loss'], state['epoch'])
            self.writer.add_scalar("01-Train-Loss/MMD-Train-Share", state['scalars']['mmd-train-share'], state['epoch'])
            self.writer.add_scalar("02-Validation-Loss/MMD-Validation-Loss", state['scalars']['mmd-val-loss'], state['epoch'])
            self.writer.add_scalar("02-Validation-Loss/CE-Validation-Loss", state['scalars']['ce-val-loss'], state['epoch'])
            self.writer.add_scalar("02-Validation-Loss/MMD-Validation-Share", state['scalars']['mmd-val-share'], state['epoch'])
            ## Kappa
            self.writer.add_scalar("00-Kappa", state['kappa'], state['epoch'])

        ## Accuracies
        self.writer.add_scalar("03-Validation-Accuracy/Accuracy", state['scalars']['val-acc'], state['epoch'])

        if self.write_embeddings:
        # Latent Representation
            if state['epoch'] in [1,5,6,10,20,30,40,50,75,100,150,200,250,300]:
                self.writer.add_embedding(state['embeddings']['z-train'], state['embeddings']['d-train'], global_step=state['epoch'], tag='Train-Data')
                self.writer.add_embedding(state['embeddings']['z-val'], state['embeddings']['d-val'], global_step=state['epoch'], tag='Validation-Data')

        ## Classification Report
        targets = ['Negative', 'Neutral', 'Positive']
        for target in targets:
            self.writer.add_scalar("04-Classification-Report/%s/Precision"%target, state['classification-report'][target]['precision'], state['epoch'])
            self.writer.add_scalar("04-Classification-Report/%s/Recall"%target, state['classification-report'][target]['recall'], state['epoch'])
            self.writer.add_scalar("04-Classification-Report/%s/F1-Score"%target, state['classification-report'][target]['f1-score'], state['epoch'])

        ## Layer Histograms
        for name, params in state['models']['model'].named_parameters():
            self.writer.add_histogram(name, torch.histc(params), state['epoch'])

        ## Domain Invariance
        if not(self.train_mode=='single-source'):
            for key in state['domain-invariance'].keys():
                self.writer.add_scalar(f"06-Domain-Invariance/{key}", state['domain-invariance'][key], state['epoch'])


        self.writer.flush()

    def write_test_results(self, state):
        self.writer.add_hparams(state['hyperparams'], state['scalars'], run_name=f"test-{self.run_name}")
        self.writer.flush()

    #def __del__(self):
    #    self.writer.flush()
    #    self.writer.close()

class print_logger():
    def __init__(self, train_mode):
        self.train_mode = train_mode
        #print("Hi - I am going to handle your Print Logs")

    def write_state(self, state):
        if self.train_mode == 'mmd':
            print("[%s] Epoch %i: - kappa: %4.2f - Total-Train-Loss: %4.2f - CLA-Train-Loss: %4.2f - MMD-Train-Loss: %4.2f - Total-Val-Loss: %4.2f - CLA-Val-Loss: %4.2f - MMD-Validation-Loss: %4.2f - Val-Accuracy: %4.2f - Elapsed Time: %4.2f s"%(
            state['run-name'], state['epoch'], state['kappa'], state['scalars']['total-train-loss'], state['scalars']['ce-train-loss'], state['scalars']['mmd-train-loss'], state['scalars']['total-val-loss'], state['scalars']['ce-val-loss'], state['scalars']['mmd-val-loss'], state['scalars']['val-acc'], state['end-time']-state['start-time']))
            print(f"[{state['run-name']}] Train-SVM-Accuracy: {state['domain-invariance']['train_svm_acc']:.2f} - Train-LSVM-Accuracy: {state['domain-invariance']['train_linear_svm_acc']:.2f} - Train-NB-Accuracy: {state['domain-invariance']['train_nb_acc']:.2f} - Train-XGB-Accuracy: {state['domain-invariance']['train_xgb_acc']:.2f} - Train-LDA-Accuracy: {state['domain-invariance']['train_lda_acc']:.2f}") 
            print(f"[{state['run-name']}] Val-SVM-Accuracy: {state['domain-invariance']['val_svm_acc']:.2f} - Val-LSVM-Accuracy: {state['domain-invariance']['val_linear_svm_acc']:.2f} - Val-NB-Accuracy: {state['domain-invariance']['val_nb_acc']:.2f} - Val-XGB-Accuracy: {state['domain-invariance']['val_xgb_acc']:.2f} - Val-LDA-Accuracy: {state['domain-invariance']['val_lda_acc']:.2f}") 

        elif self.train_mode == 'standard' or self.train_mode=='single-source':
            print("[%s] Epoch %i: - Train-Loss: %4.2f - Val-Loss: %4.2f - Val-Accuracy: %4.2f - Elapsed Time: %4.2f s"
        %(state['run-name'], state['epoch'], state['scalars']['total-train-loss'], state['scalars']['total-val-loss'], state['scalars']['val-acc'], state['end-time']-state['start-time']))
        else:
            raise NotImplementedError

    def write_test_results(self, state):
        if self.train_mode == 'adversarial':
            raise NotImplementedError
        elif self.train_mode == 'standard' or self.train_mode=='mmd':
            print("[%s] Test-Accuracy: %4.2f - SVM-Accuracy: %4.2f -LSVM-Accuracy: %4.2f - NB-Accuracy: %4.2f - XGB-Accuracy: %4.2f - LDA-Accuracy: %4.2f"
            %(state['run-name'], state['scalars']['05-Scores/test-acc'], state['scalars']['05-Scores/svm-acc'], state['scalars']['05-Scores/lsvm-acc'], state['scalars']['05-Scores/nb-acc'], state['scalars']['05-Scores/xgb-acc'], state['scalars']['05-Scores/lda-acc']))
        elif self.train_mode == 'single-source':
            print("[%s] Test-Accuracy: %4.2f"%(state['run-name'], state['scalars']['05-Scores/test-acc']))

class csv_logger():
    def __init__(self, logpath, train_mode):
        self.train_mode = train_mode
        self.logpath = logpath
        self.header_written = False
        #print("Hi - I am going to handle your CSV Logs")


    def write_train_header(self):
        # Prepare Logging
        if not (os.path.isdir(self.logpath)):
            os.makedirs(os.path.join(self.logpath))
        elif os.path.isfile(os.path.join(self.logpath, 'logs.csv')):
            print("LogFile already exists! Rename or remove it, and restart the training")
            raise Exception
        
        if self.train_mode == 'mmd':
            header = ['Epoch', 'kappa', 'Total-Train-Loss', 'CLA-Train-Loss', 'MMD-Train-Loss', 'Total-Validation-Loss', 'CLA-Validation-Loss', 'MMD-Validation-Loss', 'Validation-Accuracy']
        elif self.train_mode == 'standard' or self.train_mode == 'single-source':
            header = ['Epoch', 'Train-Loss', 'Validation-Loss', 'Validation-Accuracy']
        else:
            raise NotImplementedError
        
        with open(os.path.join(self.logpath, 'logs.csv'), 'w', encoding='UTF8') as f:
                writer = csv.writer(f)
                writer.writerow(header)

    def write_domain_invariance_header(self):
        # Prepare Logging
        if not (os.path.isdir(self.logpath)):
            os.makedirs(os.path.join(self.logpath))
        elif os.path.isfile(os.path.join(self.logpath, 'domain_invariance.csv')):
            print("LogFile already exists! Rename or remove it, and restart the training")
            raise NameError
        
        if self.train_mode == 'mmd' or self.train_mode == 'standard':
            header = ['Epoch', 'Train-SVM-Acc', 'Train-LSVM-Acc', 'Train-NB-Acc', 'Train-XGB-Acc', 'Train-LDA-ACC', 'Val-SVM-Acc', 'Val-LSVM-Acc', 'Val-NB-Acc', 'Val-XGB-Acc', 'Val-LDA-ACC']
            with open(os.path.join(self.logpath, 'domain_invariance.csv'), 'w', encoding='UTF8') as f:
                writer = csv.writer(f)
                writer.writerow(header)
        elif self.train_mode == 'adversarial':
            raise NotImplementedError

    
    def write_state(self, state):
        if not(self.header_written):
            self.write_train_header()
            self.write_domain_invariance_header()
            self.header_written = True
        with open(os.path.join(self.logpath, 'logs.csv'), 'a', encoding='UTF8') as f:
            writer = csv.writer(f)
            if self.train_mode == 'mmd':
                writer.writerow([str(state['epoch']), str(state['kappa']), str(state['scalars']['total-train-loss']), str(state['scalars']['ce-train-loss']), str(state['scalars']['mmd-train-loss']), str(state['scalars']['total-val-loss']), str(state['scalars']['ce-val-loss']), str(state['scalars']['mmd-val-loss']), str(state['scalars']['val-acc'])])
            elif self.train_mode == 'standard' or self.train_mode == 'single-source':
                writer.writerow([str(state['epoch']), str(state['scalars']['total-train-loss']), state['scalars']['total-val-loss'], str(state['scalars']['val-acc'])])
            else:
                raise NotImplementedError
        
        if self.train_mode == 'mmd' or self.train_mode == 'standard':
            with open(os.path.join(self.logpath, 'domain_invariance.csv'), 'a', encoding='UTF8') as f:
                writer = csv.writer(f)
                writer.writerow([str(state['epoch']),
                str(state['domain-invariance']['train_svm_acc']),
                str(state['domain-invariance']['train_linear_svm_acc']),
                str(state['domain-invariance']['train_nb_acc']),
                str(state['domain-invariance']['train_xgb_acc']),
                str(state['domain-invariance']['train_lda_acc']),
                str(state['domain-invariance']['val_svm_acc']),
                str(state['domain-invariance']['val_linear_svm_acc']),
                str(state['domain-invariance']['val_nb_acc']),
                str(state['domain-invariance']['val_xgb_acc']),
                str(state['domain-invariance']['val_lda_acc']),
                ])
        elif self.train_mode == 'adversarial':
            raise NotImplementedError

    def write_test_results(self, state):
        if self.train_mode == 'adversarial':
            raise NotImplementedError
        elif self.train_mode == 'standard' or self.train_mode=='mmd':
            header = ['Test-Accuracy', 'SVM-Accuracy', 'Linear-SVM-Accuracy', 'NB-Accuracy', 'XGB-Accuracy', 'LDA-Accuracy']
            row = [str(state['scalars']['05-Scores/test-acc']), str(state['scalars']['05-Scores/svm-acc']), str(state['scalars']['05-Scores/lsvm-acc']), str(state['scalars']['05-Scores/nb-acc']), str(state['scalars']['05-Scores/xgb-acc']), str(state['scalars']['05-Scores/lda-acc'])]
        elif self.train_mode == 'single-source':
            header = ['Test-Accuracy']
            row = [str(state['scalars']['05-Scores/test-acc'])]

        with open(os.path.join(self.logpath, 'test_logs.csv'), 'a', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerow(row)