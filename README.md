# DAPE Framework
DAPE (Domain Aligned Private Encoders) is a Framework for learning from multiple data-sources. Thereby, multiple encoders are trained using one common classifier. An MMD Loss is used for aligning the encoders during train-time.

## Description
The DAPE Framework consists of mutliple encoders, one for each-data source. No matter which encoder is used, it is even possible to use different encoders for different data-sources. The only constrain is, that the output of each encoder has to have the same shape. From the latent representations that the encoders output, a shared classifier is making predictions for the samples of each data-source. A visualization of the Framework is shown in ![DAPE Framework Architecture](img/DAPE_Architecture.pdf).

## How to use it
In order to train the Framework, you can use the methods in `pipeline_funcs.py`, i.e. `train`, `test`, etc. to build yourself a pipeline, or you can use the method `pipeline`in the file `pipeline.py` to train and test a DAPE Framework. For more information on the parameters you need to pass to the method, please refer to the comments in the file.
