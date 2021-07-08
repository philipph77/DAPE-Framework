def train(enc, cla, X, Y, batch_size, epochs=500, early_stopping_after_epochs=50):
    import math
    import numpy as np
    from tqdm import trange
    import time
    from sklearn.utils import shuffle
    import torch
    import torch.optim as optim
    import torch.nn as nn
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.metrics import accuracy_score

    optimizer = optim.SGD([{'params': enc.parameters()},{'params': cla.parameters()}], lr=0.01)
    criterion = nn.CrossEntropyLoss()
    X, Y = shuffle(X,Y, random_state=7)
    ohe = OneHotEncoder(sparse=False)
    #Y = torch.from_numpy(ohe.fit_transform(Y.reshape(-1,1))).float()
    num_batches = math.ceil(float(X.shape[0]) / batch_size)
    batch_start_idx = np.arange(0, num_batches*batch_size, batch_size)
    batch_end_idx = batch_start_idx + batch_size
    batch_end_idx[-1] = X.shape[0]

    assert len(batch_start_idx) == num_batches
    assert len(batch_end_idx) == num_batches

    min_loss = np.infty

    for epoch in range(epochs):
        start_time = time.time()
        acc = 0.
        loss = 0.
        for i in trange(num_batches):
            optimizer.zero_grad()
            x_batch = X[batch_start_idx[i]:batch_end_idx[i],:,:].unsqueeze_(1)
            y_batch = Y[batch_start_idx[i]:batch_end_idx[i]]
            y_pred = cla(enc(x_batch)).squeeze_()
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            acc = acc + accuracy_score(y_batch.detach().numpy(), np.argmax(y_pred.detach().numpy(),axis=1))

        # This line is to handle the fact that the last batch may be smaller then the other batches
        acc = acc - accuracy_score(y_batch.detach().numpy(), np.argmax(y_pred.detach().numpy(),axis=1)) + float(batch_end_idx[-1]-batch_start_idx[-1])/batch_size  * accuracy_score(y_batch.detach().numpy(), np.argmax(y_pred.detach().numpy(),axis=1))

        acc = acc / (float(x_list[0].shape[0]) / batch_size)
        
        end_time = time.time()

        print("Epoch %i: - Loss: %4.2f - Accuracy: %4.2f - Elapsed Time: %4.2f s"%(epoch, loss, acc, end_time-start_time))





if __name__ == '__main__':
    import torch
    import numpy as np
    import architectures
    from torchinfo import summary

    C = 32
    T = 2*128
    F1 = 32
    D = 16
    F2 = 8
    enc = architectures.EEGNetEncoder(channels=C, temporal_filters=F1, spatial_filters=D, pointwise_filters=F2, dropout_propability=0.25, latent_dim=20)
    cla = architectures.DenseClassifier(20,3, max_norm=0.25)
    #enc = architectures.DeepConvNetEncoder(channels=C, latent_dim=20)
    #cla = architectures.DenseClassifier(20,3, max_norm=False, use_bias=False)
    dataset = np.load('../../Datasets/private_encs/DEAP.npz')
    X = torch.from_numpy(dataset['X']).type(torch.FloatTensor)
    Y = torch.from_numpy(dataset['Y']).type(torch.LongTensor) + 1

    batch_size = 64

    summary(enc,(batch_size, 1,32,256))
    summary(cla, (batch_size, 1,20))

    train(enc, cla, X, Y, batch_size = batch_size)