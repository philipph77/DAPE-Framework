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

def send_mail_notification(run_name):
    try:
        import smtplib
        SUBJECT = 'Run Fertig'
        TEXT = 'Es ist ein Fehler aufgetreten bei: ' + run_name
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

if __name__=='__main__':
    pass
    

