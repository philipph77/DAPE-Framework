def main():
    import torch
    import numpy as np
    import architectures
    from torchsummary import summary

    
    batch_size = 4
    latent_dim = 20

    mydict = {
        0: np.random.randn(batch_size, latent_dim),
        1: np.random.randn(batch_size, latent_dim),
        2: np.random.randn(batch_size, latent_dim),
        3: np.random.randn(batch_size, latent_dim),
    }

    Z = mydict[0]
    for datasource_idx in range(1,len(mydict)): 
        Z = np.concatenate((Z, mydict[datasource_idx]), axis=0)

if __name__ == '__main__':
    import numpy as np
    list1 = [3,7,5,1,3]
    list2 = [2,3,5]
    list3 = [1,6,7]

    wrapper = list()
    
    wrapper.append(list1)

    wrapper.append(list2)

    wrapper.append(list3)

    wrapper.append(np.arange(0, 12*4, 4))

    print(wrapper)
        