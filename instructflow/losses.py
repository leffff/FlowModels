import torch
from torch.nn import functional as F
from torch.distributions.normal import Normal


def flow_matching_loss(model, x_0, x_1, t):
    t = t.reshape((t.shape[0], 1, 1, 1))
    
    x_t = t * x_1 + (1 - t) * x_0
    t = t.flatten()
    noise_pred = model(x_t, t)
    
    return F.mse_loss(noise_pred, (x_1 - x_0), reduction="mean")


def bridge_matching_loss(model, x_0, x_1, t):
    t = t.reshape((t.shape[0], 1, 1, 1))

    sigma = 1e-2
    x_t = Normal(
              loc=(t * x_1 + (1 - t) * x_0),
              scale=(sigma * t * (1 - t) * torch.ones_like(x_1)) + 1e-8
          ).sample()
    t = t.flatten()
    
    noise_pred = model(x_t, t)

    t = t.reshape((t.shape[0], 1, 1, 1))
    return F.mse_loss(noise_pred, (x_1 - x_0) / (1 - t + 1e-8), reduction="mean")