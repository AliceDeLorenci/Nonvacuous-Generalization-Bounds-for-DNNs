# Nonvacuous-Generalization-Bounds-for-DNNs

|     Experiment     | Train Error | Test Error | SNN Train Error | SNN Test Error | PAC-Bayes Bound |
|:-----------------:|:-----------:|:----------:|:---------------:|:--------------:|:---------------:|
| $600^2$      |   0.006      |   0.019     |      0.389       |      0.389      |      0.521       |
| $600^2$ (with $1e^{-3}$ weight decay)      |   0.005      |   0.018     |      0.501      |      0.501      |      0.635       |

**obs. 1:** Ridge-regularized model was trainned for 30 epochs to achieve accuracy comparable to that of the non-regulaarized model, which was trained for 20 epochs.

**obs. 2:** Our SNN errors are much larger than those reported by Dziugaite et al., maybe because they trained the SNNs for many more epochs than us.