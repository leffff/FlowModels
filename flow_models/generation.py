import torch
from torch.distributions.normal import Normal


def generate(model, vae, x_0, encoder_hidden_states, n_steps: int = 100, device: str = "cuda"):
    model.to(device)
    vae.to(device)
    model.eval()
    vae.eval()
    
    x_t = x_0.to(device)

    bs = x_0.shape[0]
    
    eps = 1e-8
    t = torch.linspace(eps, 1 - eps, n_steps + 1).to(device)

    for i in range(1, len(t)):
        t_prev = t[i - 1].unsqueeze(0).repeat((bs,))

        with torch.no_grad():
            f_eval = model(x_t, t_prev, encoder_hidden_states=encoder_hidden_states).sample
            
        x_t = x_t + (t[i] - t[i - 1]) * f_eval

    with torch.no_grad():
        x_t = vae.decode(x_t / 0.18215).sample
    
    return x_t


def generate_euler(model, vae, x_0, encoder_hidden_states, n_steps: int = 100, device: str = "cuda"):
    model.to(device)
    vae.to(device)
    model.eval()
    vae.eval()
    
    x_t = x_0.to(device)

    bs = x_0.shape[0]
    
    eps = 1e-8
    t = torch.linspace(eps, 1 - eps, n_steps + 1).to(device)
    h = 1 / n_steps
    
    for i in range(1, len(t)):
        t_prev = t[i - 1].unsqueeze(0).repeat((bs,))

        with torch.no_grad():
            k1 = model(x_t, t_prev, encoder_hidden_states=encoder_hidden_states).sample
            
        x_t = x_t + h * k1

    with torch.no_grad():
        x_t = vae.decode(x_t / 0.18215).sample
    
    return x_t


def generate_midpoint(model, vae, x_0, encoder_hidden_states, n_steps: int = 100, device: str = "cuda"):
    model.to(device)
    vae.to(device)
    model.eval()
    vae.eval()
    
    x_t = x_0.to(device)

    bs = x_0.shape[0]
    
    eps = 1e-8
    t = torch.linspace(eps, 1 - eps, n_steps + 1).to(device)
    h = 1 / n_steps

    for i in range(1, len(t)):
        t_prev = t[i - 1].unsqueeze(0).repeat((bs,))

        with torch.no_grad():
            k1 = model(x_t, t_prev, encoder_hidden_states=encoder_hidden_states).sample
            k2 = model(x_t + (h / 2) * k1 , t_prev + h / 2, encoder_hidden_states=encoder_hidden_states).sample
            
        x_t = x_t + h * k2

    with torch.no_grad():
        x_t = vae.decode(x_t / 0.18215).sample
    
    return x_t


def generate_rk4(model, vae, x_0, encoder_hidden_states, n_steps: int = 100, device: str = "cuda"):
    model.to(device)
    vae.to(device)
    model.eval()
    vae.eval()
    
    x_t = x_0.to(device)

    bs = x_0.shape[0]
    
    eps = 1e-8
    t = torch.linspace(eps, 1 - eps, n_steps + 1).to(device)
    h = 1 / n_steps

    for i in range(1, len(t)):
        t_prev = t[i - 1].unsqueeze(0).repeat((bs,))

        with torch.no_grad():
            k1 = model(x_t, t_prev, encoder_hidden_states=encoder_hidden_states).sample
            k2 = model(x_t + (h / 2) * k1 , t_prev + h / 2, encoder_hidden_states=encoder_hidden_states).sample
            k3 = model(x_t + (h / 2) * k2 , t_prev + h / 2, encoder_hidden_states=encoder_hidden_states).sample
            k4 = model(x_t + h * k3 , t_prev + h, encoder_hidden_states=encoder_hidden_states).sample

        x_t = x_t + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    with torch.no_grad():
        x_t = vae.decode(x_t / 0.18215).sample
    
    return x_t


def generate_euler_cfg(model, vae, x_0, encoder_hidden_states, null_encoder_hidden_states, guidance_scale, n_steps: int = 100, device: str = "cuda"):
    model.to(device)
    model.eval()    

    x_t = x_0.to(device)

    bs = x_0.shape[0]
    
    eps = 1e-8
    t = torch.linspace(eps, 1 - eps, n_steps + 1).to(device)
    h = 1 / n_steps 
    
    for i in range(1, len(t)):
        t_prev = t[i - 1].unsqueeze(0).repeat((bs,))
        with torch.no_grad():
            x_t_cond, x_t_uncond = model(
                      torch.cat([x_t, x_t], dim=0),
                      torch.cat([t_prev, t_prev], dim=0), 
                      encoder_hidden_states=torch.cat([encoder_hidden_states, null_encoder_hidden_states], dim=0)
                  ).sample.chunk(2)
            
            k1 = (1 - guidance_scale) * x_t_uncond + guidance_scale * x_t_cond
            
            x_t = x_t + h * k1
    
    with torch.no_grad():
        x_t = vae.decode(x_t / 0.18215).sample
    
    return x_t


def generate_midpoint_cfg(model, vae, x_0, encoder_hidden_states, null_encoder_hidden_states, guidance_scale, n_steps: int = 100, device: str = "cuda"):
    model.to(device)
    model.eval()    

    x_t = x_0.to(device)

    bs = x_0.shape[0]
    
    eps = 1e-8
    t = torch.linspace(eps, 1 - eps, n_steps + 1).to(device)
    h = 1 / n_steps 
    
    for i in range(1, len(t)):
        t_prev = t[i - 1].unsqueeze(0).repeat((bs,))
        
        with torch.no_grad():
            x_t_chunk = torch.cat([x_t, x_t], dim=0)
            t_prev_chunk = torch.cat([t_prev, t_prev], dim=0)
            encoder_hidden_states_chunk = torch.cat([encoder_hidden_states, null_encoder_hidden_states], dim=0)
            k1_cond, k1_uncond = model(
                      x_t_chunk,
                      t_prev_chunk, 
                      encoder_hidden_states=encoder_hidden_states_chunk
                  ).sample.chunk(2)
            
            k1 = (1 - guidance_scale) * k1_uncond + guidance_scale * k1_cond
            
            k1 = torch.cat([k1, k1], dim=0)
            k2_cond, k2_uncond = model(x_t_chunk + (h / 2) * k1 , t_prev_chunk + h / 2, encoder_hidden_states=encoder_hidden_states_chunk).sample.chunk(2)
            
            k2 = (1 - guidance_scale) * k2_uncond + guidance_scale * k2_cond
            
            x_t = x_t + h * k2
    
    with torch.no_grad():
        x_t = vae.decode(x_t / 0.18215).sample
    
    return x_t


def generate_rk4_cfg(model, vae, x_0, encoder_hidden_states, null_encoder_hidden_states, guidance_scale, n_steps: int = 100, device: str = "cuda"):
    model.to(device)
    model.eval()    

    x_t = x_0.to(device)

    bs = x_0.shape[0]
    
    eps = 1e-8
    t = torch.linspace(eps, 1 - eps, n_steps + 1).to(device)
    h = 1 / n_steps 

    encoder_hidden_states_chunk = torch.cat([encoder_hidden_states, null_encoder_hidden_states], dim=0)
    
    for i in range(1, len(t)):
        t_prev = t[i - 1].unsqueeze(0).repeat((bs,))
        
        with torch.no_grad():
            x_t_chunk = torch.cat([x_t, x_t], dim=0)
            t_prev_chunk = torch.cat([t_prev, t_prev], dim=0)
            
            k1_cond, k1_uncond = model(x_t_chunk, t_prev_chunk, encoder_hidden_states=encoder_hidden_states_chunk).sample.chunk(2)
            k1 = (1 - guidance_scale) * k1_uncond + guidance_scale * k1_cond
            k1_chunk = torch.cat([k1, k1], dim=0)
            
            k2_cond, k2_uncond = model(x_t_chunk + (h / 2) * k1_chunk , t_prev_chunk + h / 2, encoder_hidden_states=encoder_hidden_states_chunk).sample.chunk(2)
            k2 = (1 - guidance_scale) * k2_uncond + guidance_scale * k2_cond
            k2_chunk = torch.cat([k2, k2], dim=0)

            k3_cond, k3_uncond = model(x_t_chunk + (h / 2) * k2_chunk , t_prev_chunk + h / 2, encoder_hidden_states=encoder_hidden_states_chunk).sample.chunk(2)
            k3 = (1 - guidance_scale) * k3_uncond + guidance_scale * k3_cond
            k3_chunk = torch.cat([k3, k3], dim=0)

            k4_cond, k4_uncond = model(x_t_chunk + h * k3_chunk , t_prev_chunk + h, encoder_hidden_states=encoder_hidden_states_chunk).sample.chunk(2)
            k4 = (1 - guidance_scale) * k4_uncond + guidance_scale * k4_cond

            x_t = x_t + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    
    with torch.no_grad():
        x_t = vae.decode(x_t / 0.18215).sample
    
    return x_t
