def datasources_to_binary(data_sources):
    binary = ''
    possible_sources = ['SEED', 'SEED_IV', 'DEAP', 'DREAMER']
    for source in possible_sources:
        if source in data_sources:
            binary +='1'
        else:
            binary +='0'
    return binary


def filter_datasource_files(datasource_files, data_sources):
    filtered_list = list()
    for data_source in data_sources:
        data_source+= '.'
        for file in datasource_files:
            if data_source in file:
                filtered_list.append(file)
    assert len(filtered_list) == len(data_sources)

    return sorted(filtered_list)

def generate_encoder_list(encoder, latent_dim, data_source_files, **kwargs):#data_sources, encoder, latent_dim):
    import numpy as np
    encoders = list()
    for dsf in sorted(data_source_files):
        ds = np.load(dsf)
        channels = ds['X'].shape[1]
        encoders.append(encoder(channels=channels, latent_dim=latent_dim, **kwargs))
    return encoders

def generate_run_name():
    pass

def send_mail_notification(subject, run_name, error):
    try:
        import smtplib
        SUBJECT = subject
        TEXT = 'Es ist ein Fehler aufgetreten bei: ' + run_name + '\n' + str(error)
        content = 'Subject: {}\n\n{}'.format(SUBJECT, TEXT)
        mail = smtplib.SMTP('smtp.gmail.com', 587)
        mail.ehlo()
        mail.starttls()
        mail.login('notifier.finished@gmail.com', 'tHeLtOCKwenI')
        mail.sendmail('notifier.finished@gmail.com', 'philipp.hallgarten@web.de', content)
        mail.sendmail('notifier.finished@gmail.com', 'philipp.hallgarten1@porsche.de', content)
        mail.close()
    except:
        print("Mail could not be sent")

def MMD_loss(X_list ,kernel='rbf', num_choices=0, x_one_vs_all=None):
    import numpy as np
    import torch

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def calculate_MMD(x, y, kernel):
        # Reference: https://www.kaggle.com/onurtunali/maximum-mean-discrepancy
        """Emprical maximum mean discrepancy. The lower the result
        the more evidence that distributions are the same.

        Args:
            x: first sample, distribution P
            y: second sample, distribution Q
            kernel: kernel type such as "multiscale" or "rbf"
        """

        x = x.squeeze()
        y = y.squeeze()

        xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
        rx = (xx.diag().unsqueeze(0).expand_as(xx))
        ry = (yy.diag().unsqueeze(0).expand_as(yy))
        
        dxx = rx.t() + rx - 2. * xx # Used for A in (1)
        dyy = ry.t() + ry - 2. * yy # Used for B in (1)
        dxy = rx.t() + ry - 2. * zz # Used for C in (1)
        
        XX, YY, XY = (torch.zeros(xx.shape).to(device),
                    torch.zeros(xx.shape).to(device),
                    torch.zeros(xx.shape).to(device))
        
        if kernel == "multiscale":
            
            bandwidth_range = [0.2, 0.5, 0.9, 1.3]
            for a in bandwidth_range:
                XX += a**2 * (a**2 + dxx)**-1
                YY += a**2 * (a**2 + dyy)**-1
                XY += a**2 * (a**2 + dxy)**-1
                
        if kernel == "rbf":
        
            bandwidth_range = [10, 15, 20, 50]
            for a in bandwidth_range:
                XX += torch.exp(-0.5*dxx/a)
                YY += torch.exp(-0.5*dyy/a)
                XY += torch.exp(-0.5*dxy/a)

        return torch.mean(XX + YY - 2. * XY)
    
    
    num_datasources = len(X_list)
    assert num_choices <= num_datasources # maximum number of combinations, possible using this method
    rng = np.random.default_rng()
    if num_choices == 0:
        num_choices = num_datasources
    #print("Num choices: %i"%num_choices)
    mmd_loss = 0.
    k = list(rng.choice(num_datasources, num_choices, replace=False))

    if x_one_vs_all == None:
        for i in range(len(k)-1):
            mmd_loss += calculate_MMD(X_list[k[i]], X_list[k[i+1]], kernel)
        mmd_loss += calculate_MMD(X_list[k[-1]], X_list[k[0]], kernel)
    else:
        for i in range(len(k)):
            mmd_loss += calculate_MMD(x_one_vs_all, X_list[k[i]], kernel)
    
    return mmd_loss

def fit_predict_classifier(z_fit, d_fit, z_score, d_score, clf):
    clf.fit(z_fit, d_fit)
    return clf.score(z_score, d_score)


if __name__=='__main__':
    pass
    

