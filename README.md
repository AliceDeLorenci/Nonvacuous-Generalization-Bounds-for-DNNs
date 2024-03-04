# Nonvacuous-Generalization-Bounds-for-DNNs

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

${T-600^2}$ (weight decay ${10^{-3}}$)

    Namespace(nn_type='mlp', nout=1, nlayers=2, nin_channels=1, kernel_size=3, nfilters=16, nin=784, nhid=600, batch_size=100, epochs=20, lr=0.01, weight_decay=0.0001, lr2=0.001, sigma_init=1.0, T=5000, nb_snns=200, best_loss_patience=1000, scheduler='onecycle', warmup_pct=0.1, num_workers=4)

    Train error: 0.004949995577335331 Test error 0.017499991059303333
    SNN train error 0.03798141067574422 SNN test error 0.04424549515247345
    PAC-Bayes bound (before) 0.21466593798323375
    PAC-Bayes bound 0.16179496847241528