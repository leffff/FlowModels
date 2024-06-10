import wandb
from tqdm.auto import tqdm
import torch

from generation import generate


def train_epoch(unet, vae, text_encoder, dataloader, loss_function, optimizer, scheduler, device):
    unet.to(device)
    vae.to(device)
    text_encoder.to(device)
    
    unet.train()
    vae.eval()
    text_encoder.eval()

    total_loss = 0
    batch_i = 0
    for batch in tqdm(dataloader):
        x_1, x_0, input_ids, attention_mask = batch
        x_1, x_0, input_ids, attention_mask = x_1.to(device), x_0.to(device), input_ids.to(device), attention_mask.to(device)
        bs = x_1.shape[0]

        with torch.no_grad():
            x_1_latents = 0.18215 * vae.encode(x_1).latent_dist.mean
            x_0_latents = 0.18215 * vae.encode(x_0).latent_dist.mean

            encoder_hidden_states = text_encoder(input_ids=input_ids, attention_mask=attention_mask)['last_hidden_state']

        t = torch.sigmoid(torch.randn((bs,), device=device))

        loss = loss_function(unet, x_0_latents, x_1_latents, t, encoder_hidden_states)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        total_loss += loss.item() 
        batch_i += 1

        if batch_i % 5000 == 0:
            x_gen = generate(unet=unet, vae=vae, x_0=x_0_latent[:16], encoder_hidden_states=encoder_hidden_states[:16], device=device)
           
            log = {
                "loss": total_loss / 5000,
                "source_images": wandb.Image(show_images(x_0)),
                "generated_images": wandb.Image(show_images(x_gen)),
            }
            wandb.log(log)
            
            total_loss = 0

            torch.save(unet.state_dict(), f"checkpoints/inverse_v1/{batch_i}.pt")