# Nonvacuous-Generalization-Bounds-for-DNNs**

### Reproducing the paper results

| Experiments |  $600^2$  |  $1200^2$  | $300^3$ | $600^3$ | $1200^3$ |  $600^4$  |
|-------------|:-----------:|:-----:|:-----:|:-----:|:------:|:------:|
| Train error |0.007|0.006|0.008|0.006|0.005|0.005|         
| Test error  |0.021|0.019|0.020|0.018|0.018|0.018|   
| SNN train error |0.020|0.019|0.016|0.016|0.017|0.016|         
| SNN test error |0.027|0.025|0.024|0.024|0.025|0.024|         
| PAC-Bayes bound |0.102|0.108|0.095|0.100|0.105|0.095|         
| # parameters |832201|2384401|416401|1192801|3825601|1553401|         

### Multiclass

| Experiments |  $600^2$  |  $1200^2$  | $300^3$ | $600^3$ | $1200^3$ |  $600^4$  |
|-------------|:-----------:|:-----:|:-----:|:-----:|:------:|:------:|
| Train error |0.028|0.020|0.025|0.022|0.014|0.015|  
| Test error  |0.035|0.029|0.036|0.033|0.028|0.029|
| SNN train error |0.031|0.029|0.025|0.026|0.032|0.024|         
| SNN test error |0.035|0.035|0.031|0.031|0.037|0.031|      
| PAC-Bayes bound |0.134|0.133|0.129|0.131|0.146|0.131|      
| # parameters |837610|2395210|419110|1198210|3836410|1558810|      

### Weight decay

- choose one architecture and vary weigth decay parameter

| Experiments |  $600^2, \lambda=$ | $600^2, \lambda=$ | $600^2, \lambda=$ | $600^2, \lambda=$ |  $600^2, \lambda=$  |
|-------------|:-----------:|:-----:|:-----:|:-----:|:------:|
| Train error |             |       |       |       |        |         
| Test error  |             |       |       |       |        |  
| Validation error  |             |       |       |       |        |   
| SNN train error |        |       |       |       |        |         
| SNN test error |         |       |       |       |        |         
| PAC-Bayes bound |        |       |       |       |        |         
| # parameters |           |       |       |       |        |         

### Convolutional Neural Networks

Problem with multi-layer CNNs and why we cannot compare with MLPs in all parameters:

*Explanation:* Although the number of parameters might be lower than in MLPs, deeper networks CNN (>1 layers in this case) require more computational resources due to the increased number of operations performed per layer. That is during training, the network needs to store intermediate values and gradients for backpropagation. With more layers, the computational graph becomes deeper, requiring more memory to store gradients and intermediate values.



- The objective is to compare the Pac Bayes Bounds across shallow and deeper CNN architectures with matching parameters, alongside an almost matching-parameter MLP counterpart (CNN 1-layer 32 ->  MLP  2 layer 300, etc.). Specifically, we aim to assess the performance trade-offs between model complexity and generalization capacity for CNNs within the binary and multiclass setting.

- use the same number of layers and choose kernels to match activation sizes?

|  Experiments       |$12$    |$32$   |$64$    | $128$   | $64^2$  | $128^2$ |
|---------|-------|-------|-------|-------|-------|-------|
| Train error | 0.028 | 0.020 | 0.016 | 0.013 | 0.010 | 0.008 |
| Test error  | 0.033 | 0.033 | 0.024 | 0.028 | 0.016 | 0.012 |
| SNN train error | 0.040 | 0.055 | 0.033 | 0.029 | 0.024 | 0.038 |
| SNN test error  | 0.042 | 0.054 | 0.036 | 0.031 | 0.026 | 0.038 |
| PB          | 0.118 | 0.127 | 0.114 | 0.122 | 0.103 | 0.140 |
| Params      | 117397| 317537| 652737| 1378433| 99841 | 346369|

### Convolutional Neural Networks (MULTICLASS)

- use the same number of layers and choose kernels to match activation sizes?

| Experiments        | $12$    | $32$    | $64$    | $128$   | $64^2$  | $128^2$ |
|---------|-------|-------|-------|-------|-------|-------|
| Train error | 0.025 | 0.018 | 0.015 |0.013 | 0.024 | 0.015 | 
| Test error  | 0.023 | 0.020 | 0.016 |0.018 | 0.023 | 0.020 | 
| SNN train error | 0.038 | 0.031 | 0.025 | 0.022 | 0.034 | 0.023 |
| SNN test error  | 0.035 | 0.029 | 0.025 |  0.022 |0.032 | 0.023 |
| PB          | 0.115 | 0.114 | 0.106 | 0.107 | 0.124 | 0.119 |
| Params      | 120862| 321002| 656202|1381898| 103306 | 349834| 

### Preliminary results

${T-600^2}$ (same experiment as in the paper)

    Namespace(nn_type='mlp', nout=1, nlayers=2, nin_channels=1, kernel_size=3, nfilters=16, nin=784, nhid=600, batch_size=100, epochs=20, lr=0.01, weight_decay=0.0, lr2=0.001, sigma_init=1.0, T=5000, nb_snns=200, best_loss_patience=1000, scheduler='onecycle', warmup_pct=0.1, num_workers=4)

    Train error: 0.0050333287318548026 Test error 0.01899999201297764
    SNN train error 0.030377242399255433 SNN test error 0.03666199538707733
    PAC-Bayes bound (before) 0.2086889049924911
    PAC-Bayes bound 0.15427034505947754


${T-600^2}$ (multiclass)

    Namespace(nn_type='mlp', nout=10, nlayers=2, nin_channels=1, kernel_size=3, nfilters=16, nin=784, nhid=600, batch_size=100, epochs=20, lr=0.01, weight_decay=0.0, lr2=0.001, sigma_init=1.0, T=5000, nb_snns=200, best_loss_patience=1000, scheduler='onecycle', warmup_pct=0.1, num_workers=4)
    
    Number of parameters: 1198210
    Train error: 0.02041665653387703 Test error 0.03559999525547031
    SNN train error 0.03273366013616323 SNN test error 0.03727699551582336
    PAC-Bayes bound (before) 0.16579863788798294
    PAC-Bayes bound 0.12219981669868166

${T-600^2}$ (weight decay ${10^{-3}}$)

    Namespace(nn_type='mlp', nout=1, nlayers=2, nin_channels=1, kernel_size=3, nfilters=16, nin=784, nhid=600, batch_size=100, epochs=20, lr=0.01, weight_decay=0.001, lr2=0.001, sigma_init=1.0, T=5000, nb_snns=200, best_loss_patience=1000, scheduler='onecycle', warmup_pct=0.1, num_workers=4)

    Number of parameters: 1192801
    Train error: 0.007299993634223956 Test error 0.017999992370605455
    SNN train error 0.03443924336532751 SNN test error 0.04124349569082261
    PAC-Bayes bound (before) 0.1862028275869163
    PAC-Bayes bound 0.13795737911536815

${T-600^2}$ (weight decay ${10^{-4}}$)

    Namespace(nn_type='mlp', nout=1, nlayers=2, nin_channels=1, kernel_size=3, nfilters=16, nin=784, nhid=600, batch_size=100, epochs=20, lr=0.01, weight_decay=0.0001, lr2=0.001, sigma_init=1.0, T=5000, nb_snns=200, best_loss_patience=1000, scheduler='onecycle', warmup_pct=0.1, num_workers=4)

    Train error: 0.004949995577335331 Test error 0.017499991059303333
    SNN train error 0.03798141067574422 SNN test error 0.04424549515247345
    PAC-Bayes bound (before) 0.21466593798323375
    PAC-Bayes bound 0.16179496847241528
