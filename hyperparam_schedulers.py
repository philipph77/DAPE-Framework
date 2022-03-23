import numpy as np

class hyperparam_schedule():
    def __init__(self):
        pass

    def __getitem__(self):
        pass

    def __str__(self):
        pass

class constant_schedule(hyperparam_schedule):
    def __init__(self, value=1):
        self.value = value

    def __getitem__(self, epoch):
        return self.value

    def __str__(self):
        return f"C-{self.value}"

class constant_linear_schedule(hyperparam_schedule):
    def __init__(self, start_epoch=0, start_value=0, step_value=0.5):
        self.start_epoch = start_epoch
        self.start_value = start_value
        self.step_value = step_value
    
    def __getitem__(self, epoch):
        if epoch <= self.start_epoch:
            return self.start_value
        else:
            return self.start_value + (epoch-self.start_epoch)*self.step_value

    def __str__(self):
        return f"CL-{self.start_epoch}-{self.start_value}-{self.step_value}"


class linear_constant_schedule(hyperparam_schedule):
    def __init__(self, step_value=0.1, stop_epoch=10):
        self.step_value = step_value
        self.stop_epoch = stop_epoch

    def __getitem__(self,epoch):
        if epoch < self.stop_epoch:
            return epoch * self.step_value
        else:
            return self.stop_epoch * self.step_value

    def __str__(self):
        return f"LC-{self.step_value}-{self.stop_epoch}"

class constant_linear_constant_schedule(hyperparam_schedule):
    def __init__(self, start_epoch=0, start_value=0, step_value=0.1, stop_epoch=1):
        self.start_epoch = start_epoch
        self.start_value = start_value
        self.step_value = step_value
        self.stop_epoch = stop_epoch

    def __getitem__(self, epoch):
        if epoch < self.start_epoch:
            return self.start_value
        elif epoch < self.stop_epoch:
            return self.start_value + (epoch-self.start_epoch)*self.step_value
        else:
            return self.start_value + (self.stop_epoch-self.start_epoch)*self.step_value
    
    def __str__(self):
        return f"CLC-{self.start_epoch}-{self.start_value}-{self.step_value}-{self.stop_epoch}"

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    epochs = np.arange(20)

    c = constant_schedule(value = 1)
    cl = constant_linear_schedule(2, 0.2, 0.1)
    lc = linear_constant_schedule(0.5, 10)
    clc = constant_linear_constant_schedule(4, 0.4, 0.05, 17)


    constant_values = list()
    constant_linear_values = list()
    linear_constant_values = list()
    constant_linear_constant_values = list()
    for epoch in epochs:
        constant_values.append(c[epoch])
        constant_linear_values.append(cl[epoch])
        linear_constant_values.append(lc[epoch])
        constant_linear_constant_values.append(clc[epoch])
    plt.plot(epochs, constant_values, label='Constant Schedule')
    plt.plot(epochs, constant_linear_values, label='Constant Linear Schedule')
    plt.plot(epochs, linear_constant_values, label='Linear Constant Schedule')
    plt.plot(epochs, constant_linear_constant_values, label='Constant Linear Constant Schedule')
    plt.legend()
    plt.title('Overview over different Schedulers')
    plt.show()