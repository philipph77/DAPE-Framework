import os
import numpy as np
from tqdm import tqdm, trange

def create_splits(path):
    examples = list()
    names = list()
    datasets = list()
    files = [f for f in sorted(os.listdir(path)) if f.endswith('.npz') and not ('split' in f) and not ('test' in f)]
    print(files)
    for file in tqdm(files):
        names.append(file)
        datasets.append(dict(np.load(os.path.join(path,file), allow_pickle=True)))
        examples.append(datasets[-1]['X'].shape[0])
    
    for datasource in trange(len(names)):
        datasets[datasource]['X'] = datasets[datasource]['X'][:min(examples),:,:]
        datasets[datasource]['Y'] = datasets[datasource]['Y'][:min(examples)]
        print(names[datasource])
        print(datasets[datasource]['X'].shape)
        np.savez_compressed(os.path.join(path, 'split_' + names[datasource]), **datasets[datasource])

    

if __name__ == '__main__':
    path = '../../Datasets/private_encs/'
    examples = [12834]
    #create_splits(path)
    files = [f for f in sorted(os.listdir(path)) if f.endswith('.npz') and not ('split' in f) and not ('test' in f)]
    for file in tqdm(files):
        print(file)
        dataset = np.load(os.path.join(path, file))
        datasplit = np.load(os.path.join(path, 'split_'+file))
        assert (dataset['X'][:min(examples),:,:] == datasplit['X']).all()
        assert (dataset['Y'][:min(examples)] == datasplit['Y']).all()
        assert datasplit['X'].shape[0] == datasplit['Y'].shape[0]