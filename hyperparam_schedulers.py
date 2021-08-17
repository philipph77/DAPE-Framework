import numpy as np

def constant_schedule(epoch, value=1):
    return value

def constant_linear_schedule(epoch, start_epoch=0, start_value=0, step_value=0.5):
    if epoch <= start_epoch:
        return start_value
    else:
        return start_value + (epoch-start_epoch)*step_value

def linear_constant_schedule(epoch, step_value=0.1, stop_epoch=10):
    if epoch < stop_epoch:
        return epoch * step_value
    else:
        return stop_epoch * step_value

def constant_linear_constant_schedule(epoch, start_epoch=0, start_value=0, step_value=0.1, stop_epoch=1):
    if epoch < start_epoch:
        return start_value
    elif epoch < stop_epoch:
        return start_value + (epoch-start_epoch)*step_value
    else:
        return start_value + (stop_epoch-start_epoch)*step_value

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    epochs = np.arange(20)
    constant_values = list()
    constant_linear_values = list()
    linear_constant_values = list()
    constant_linear_constant_values = list()
    for epoch in epochs:
        constant_values.append(constant_schedule(epoch, 1))
        constant_linear_values.append(constant_linear_schedule(epoch, 2, 0.2, 0.1))
        linear_constant_values.append(linear_constant_schedule(epoch, 0.5, 10))
        constant_linear_constant_values.append(constant_linear_constant_schedule(epoch, 4, 0.4, 0.05, 17))
    plt.plot(epochs, constant_values, label='Constant Schedule')
    plt.plot(epochs, constant_linear_values, label='Constant Linear Schedule')
    plt.plot(epochs, linear_constant_values, label='Linear Constant Schedule')
    plt.plot(epochs, constant_linear_constant_values, label='Constant Linear Constant Schedule')
    plt.legend()
    plt.title('Overview over different Schedulers')
    plt.show()