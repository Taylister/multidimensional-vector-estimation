from torch.nn.functional import mse_loss
import torch

def estimate_network_loss(network_output, latent_variable):
    rmse_loss = torch.sqrt(mse_loss(network_output, latent_variable))
    return rmse_loss