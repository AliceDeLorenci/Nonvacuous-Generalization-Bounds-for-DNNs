# Nonvacuous-Generalization-Bounds-for-DNNs

This repository contains a PyTorch implementation of the PAC-Bayes bound optimization algorithm introduced by the paper:

> Gintare Karolina Dziugaite and Daniel M. Roy. *Computing nonvacuous generalization bounds for deep (stochastic) neural networks with many more parameters than training data.* In Proceedings of the 33rd Annual Conference on Uncertainty in Artificial Intelligence (UAI), 2017.

## Instructions to run the code

In order to run the code, the following packages, listed on `requirements.txt`must be installed:
```
torch
torchvision
torchmetrics
matplotlib
tqdm
numpy
pandas
```

Run the code with:
```bash
python3 main.py
```

Detailed information on command line options can be obtained with:
```bash
python3 main.py -h
```

Essentially, the following parameters may be specified:
- `--nn_type`: type of neural network to use
- `--nout`: number of outputs
- `--nlayers`: number of hidden layers in the networks
- `--nin_channels`: number of input channels (when using CNNs)
- `--kernel_size`: size of the convolutional kernel (when using CNNs)
- `--nfilters`: number of filters in the convolutional layers (when using CNNs)
- `--nin`: input dimension
- `--nhid`: number of hidden units in a hidden layer (when using FCNs)
- `--batch_size`: batch size for SGD optimization
- `--epochs`: number of epochs for SGD optimization
- `--lr`: learning rate for SGD optimization
- `--weight_decay`: weight decay for SGD optimization
- `--lr2`: learning rate for PAC-Bayes bound optimization
- `--sigma_init`: scaling to apply to the initial value of s for PAC-Bayes bound optimization
- `--T`: mumber of iterations for PAC-Bayes bound optimization
- `--nb_snns`: number of SNNs to sample for MC approximation
- `--best_loss_patience`: patience of 2nd loop best loss
- `--scheduler`: scheduler for the learning rate for PAC-Bayes bound optimization
- `--warmup_pct`: percentage of iterations to warm up for PAC-Bayes bound optimization
- `--use_validation`: whether to use validation set during training
    

All execution artifacts are stored in a timestamped subdirectory inside the `save` directory, they include:
- Experiment configurations (`args.json`)
- NN model trained with SGD (`sgd_model.pt`)
- SNN parameters resulting from the optimization of the PAC-Bayes bound (`snn_model_parameters.npz`)
- Train and test errors for the NN model trained with SGD and the SNN, as well as the PAC-Bayes bound (`results.txt`)

## Experiments

The initial SGD training was carried out over ${20}$ epochs, using a batch size of ${100}$, learning rate ${\gamma = 0.01}$ and a ${\mu=0.9}$ momentum factor. Meanwhile, the optimization of the PAC-Bayes bound was carried out over ${1000}-2000$ epochs, using the RMSprop algorithm and a single cycle cosine learning rate schedule, with max learning rate of ${0.001}$ and $5\%$ of warmup epochs. As in the original paper, we use $\delta= 0.025, \delta'=0.01, b =100, c= 0.1$. Finally, we used ${n=200}$ samples of ${Q}$ to compute the Monte Carlo approximation $\hat Q_n$.

The mean ${w}$ of the posterior ${Q}$ is initialized using the neural network trained by SGD, while the diagonal ${s}$ of the covariance matrix ${\text{diag}(s)}$ is initialized to ${|w|}$ and ${\lambda}$ is initialized as ${e^{-6}}$. The prior mean is fixed to a randomly sampled ${w_0}$.

For training Convolutional Neural Networks (CNNs), we employed zero padding, and \(3 \times 3\) convolutional filters, $k$. Moreover, we integrated two linear layers with $384$ neurons serving as intermediate fully connected neurons. The decision regarding the number of convolutional layers, $l$, and the number of filters, $k$, was guided by our intention to match the number of parameters with those of Multi-Layer Perceptrons (MLPs).

Although CNNs generally have fewer parameters than MLPs, deeper networks (\(l>1\) in this case) require more computational resources due to the increased number of operations per layer. This is because during training, the network must store intermediate values and gradients for backpropagation. In essence, as the number of layers increases, the computational graph becomes deeper, necessitating more memory. Considering this, we opt for a specific choice of filter sizes: \(k = \{12, 32, 64, 128\}_{l=1}\) and \(k = \{64, 128\}_{l=2}\).


### Binary MNIST with FCNs

| Experiments | $600$ | $1200$ |$300^2$   | $600^2$  |  $1200^2$  | $300^3$ | $600^3$ | $1200^3$ |  $600^4$  |
|-------------|:----------:|:----------:|:----------:|:-----------:|:-----:|:-----:|:-----:|:------:|:------:|
| Train error | 0.014| 0.012| 0.009|0.007|0.006|0.008|0.006|0.005|0.005|         
| Test error  | 0.023| 0.021| 0.019|0.021|0.019|0.020|0.018|0.018|0.018|   
| SNN train error | 0.024 | 0.025| 0.019|0.020|0.019|0.016|0.016|0.017|0.016|         
| SNN test error | 0.030| 0.030 | 0.026|0.027|0.025|0.024|0.024|0.025|0.024|         
| PAC-Bayes bound | 0.103| 0.107| 0.094|0.102|0.108|0.095|0.100|0.105|0.095|         
| # parameters | 471601| 943201| 326101|832201|2384401|416401|1192801|3825601|1553401|         

### Multiclass MNIST with FCNs

| Experiments | $600$ | $1200$ |$300^2$   | $600^2$  |  $1200^2$  | $300^3$ | $600^3$ | $1200^3$ |  $600^4$  |
|-------------|:----------:|:----------:|:----------:|:-----------:|:-----:|:-----:|:-----:|:------:|:------:|
| Train error | 0.045| 0.040| 0.038|0.028|0.020|0.025|0.022|0.014|0.015|  
| Test error  | 0.050 | 0.045| 0.043|0.035|0.029|0.036|0.033|0.028|0.029|
| SNN train error | 0.039 | 0.040 | 0.030|0.031|0.029|0.025|0.026|0.032|0.024|         
| SNN test error | 0.041| 0.042| 0.034|0.035|0.035|0.031|0.031|0.037|0.031|      
| PAC-Bayes bound | 0.130| 0.132| 0.123|0.134|0.133|0.129|0.131|0.146|0.131|      
| # parameters | 477010| 954010| 328810 |837610|2395210|419110|1198210|3836410|1558810|      

### Binary MNIST with CNNs

The objective is to compare the Pac Bayes Bounds across shallow and deeper CNN architectures with matching parameters, alongside an almost matching-parameter MLP counterpart. Specifically, we aim to assess the performance trade-offs between model complexity and generalization capacity for CNNs within the binary and multiclass setting.


|  Experiments       |$12$    |$32$   |$64$    | $128$   | $64^2$  | $128^2$ |
|---------|-------|-------|-------|-------|-------|-------|
| Train error | 0.028 | 0.020 | 0.016 | 0.013 | 0.010 | 0.008 |
| Test error  | 0.033 | 0.033 | 0.024 | 0.028 | 0.016 | 0.012 |
| SNN train error | 0.040 | 0.055 | 0.033 | 0.029 | 0.024 | 0.038 |
| SNN test error  | 0.042 | 0.054 | 0.036 | 0.031 | 0.026 | 0.038 |
| PB          | 0.118 | 0.127 | 0.114 | 0.122 | 0.103 | 0.140 |
| Params      | 117397| 317537| 652737| 1378433| 99841 | 346369|

### Multiclass MNIST with CNNs

| Experiments        | $12$    | $32$    | $64$    | $128$   | $64^2$  | $128^2$ |
|---------|-------|-------|-------|-------|-------|-------|
| Train error | 0.025 | 0.018 | 0.015 |0.013 | 0.024 | 0.015 | 
| Test error  | 0.023 | 0.020 | 0.016 |0.018 | 0.023 | 0.020 | 
| SNN train error | 0.038 | 0.031 | 0.025 | 0.022 | 0.034 | 0.023 |
| SNN test error  | 0.035 | 0.029 | 0.025 |  0.022 |0.032 | 0.023 |
| PB          | 0.115 | 0.114 | 0.106 | 0.107 | 0.124 | 0.119 |
| Params      | 120862| 321002| 656202|1381898| 103306 | 349834| 
