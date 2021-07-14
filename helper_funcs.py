import numpy as np

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def gridsearch(model, training_data, validation_data, batch_sizes=None, num_workers=None):
    from pipeline_funcs import train
    from torch.utils.data import DataLoader
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