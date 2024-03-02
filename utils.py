import torch


def vec2params(new_params, model):
    """
    This function updates the model parameters based on the values in new_params.
    Unlike the torch implementation, this function does not mess with the gradients
    """
    l = 0
    for param in model.parameters():
        nl = param.numel()
        param = new_params[l:l+nl].reshape(param.shape)
        l += nl