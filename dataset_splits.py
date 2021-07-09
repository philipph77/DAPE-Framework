import os
import numpy as np
from tqdm import tqdm

def create_splits(path):
    shapes = list()
    names = list()
    for file in tqdm(sorted(os.listdir(path))):
        if not(file.endswith('npz')):
            continue
        names.append(file)
        dataset = np.load(os.path.join(path,file))
        shapes.append(dataset['X'].shape)
    print(names)
    print(shapes)

if __name__ == '__main__':
    #create_splits('../../Datasets/private_encs/')
    import platform
    print(platform.system())