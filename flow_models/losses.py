import torch
from torch.nn import functional as F


def flow_matching_loss(model, x_0, x_1, t, encoder_hidden_states):
    t = t.reshape((t.shape[0], 1, 1, 1))
    
    x_t = t * x_1 + (1 - t) * x_0
    t = t.flatten()

    noise_pred = model(x_t, t, encoder_hidden_states=encoder_hidden_states).sample
    
    return F.mse_loss(noise_pred, (x_1 - x_0), reduction="mean")
