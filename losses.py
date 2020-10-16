from torch.nn.functional import mse_loss

def estimate_network_loss(network_output, latent_variable):
    return mse_loss(network_output, latent_variable)