import torch
from torch.distributions.normal import Normal


def generate(model, x_0, n_steps: int = 100, device: str = "cuda"):
    model.to(device)
    model.eval()
    x_t = x_0.to(device)

    bs = x_0.shape[0]
    
    eps = 1e-8
    t = torch.linspace(eps, 1 - eps, n_steps + 1).to(device)

    for i in range(1, len(t)):
        t_prev = t[i - 1].unsqueeze(0).repeat((bs,))

        with torch.no_grad():
            f_eval = model(x_t, t_prev)
            
        x_t = x_t + (t[i] - t[i - 1]) * f_eval

    return x_t


# def generate_brown(model, x_0, n_steps: int = 100, device: str = "cuda"):
#     model.to(device)
#     model.eval()
#     x_t = x_0.to(device)

#     bs = x_0.shape[0]
    
#     eps = 1e-8
#     sigma = 1e-2
#     t = torch.linspace(eps, 1 - eps, n_steps + 1).to(device)

#     for i in range(1, len(t)):
#         t_prev = t[i - 1].unsqueeze(0).repeat((bs,))
        
#         with torch.no_grad():
#             if i == n_steps:
#                 x_t = x_t + (t[i] - t[i - 1]) * model(x_t, t_prev)
#             else:
#                 z_t = Normal(torch.zeros_like(x_t), torch.ones_like(x_t)).sample().to(device)
#                 x_t = x_t + (t[i] - t[i - 1]) * model(x_t, t_prev) + torch.sqrt(torch.tensor((sigma / n_steps), dtype=torch.float32, device=device)) * z_t

#     return x_t



def generate_brown(model, x_0, n_steps: int = 100, device: str = "cuda"):
    model.to(device)
    model.eval()
    x_t = x_0.to(device)

    bs = x_0.shape[0]
    
    eps = 1e-8
    sigma = 1e-2
    t = torch.linspace(eps, 1 - eps, n_steps + 1).to(x_0.device)

    for i in range(1, len(t)):
        t_prev = t[i - 1].unsqueeze(0)
        
        if i == n_steps:
            x_t = x_0 + (1 / n_steps) * model(x_0, t_prev)
        else:
            z_t = Normal(torch.zeros_like(x_0), torch.ones_like(x_0)).sample().to(device)
            x_t = x_0 + (1 / n_steps) * model(x_0, t_prev) + torch.sqrt(torch.tensor((sigma / n_steps), dtype=torch.float32, device=device)) * z_t
            x_0 = x_t

    return x_t
