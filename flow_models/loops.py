import wandb
from tqdm.auto import tqdm
import torch
import scipy

from flow_models.generation import generate
from flow_models.distances import l2_dist
from flow_models.utils import show_images



def train_epoch(model, vae, text_encoder, dataloader, loss_function, optimizer, scheduler, device, immiscible=False, checkpoint_path="./checkpoints", log_every_n=5000):
    model.to(device)
    vae.to(device)
    text_encoder.to(device)
    
    model.train()
    vae.eval()
    text_encoder.eval()

    total_loss = 0
    batch_i = 0
    for batch in tqdm(dataloader):
        x_1, input_ids, attention_mask = batch
        x_1, input_ids, attention_mask = x_1.to(device), input_ids.to(device), attention_mask.to(device)
        bs = x_1.shape[0]

        with torch.no_grad():
            x_1_latents = 0.18215 * vae.encode(x_1).latent_dist.mean
            encoder_hidden_states = text_encoder(input_ids=input_ids, attention_mask=attention_mask)['last_hidden_state']

        x_0_latents = torch.randn_like(x_1_latents, device=device)

        if immiscible:
            plan = scipy.optimize.linear_sum_assignment(l2_dist(x_1_latents, x_0_latents).cpu())[1]
            x_0_latents = x_0_latents[plan]
        
        t = torch.sigmoid(torch.randn((bs,), device=device))

        loss = loss_function(model, x_0_latents, x_1_latents, t, encoder_hidden_states)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        total_loss += loss.item() 
        batch_i += 1

        if batch_i % log_every_n == 0:
            x_gen = generate(model=model, vae=vae, x_0=x_0_latents[:16], encoder_hidden_states=encoder_hidden_states[:16], device=device)
           
            log = {
                "loss": total_loss / log_every_n,
                "generated_images": wandb.Image(show_images(x_gen)),
            }
            wandb.log(log)
            
            total_loss = 0

            torch.save(model.state_dict(), f"{checkpoint_path}/{batch_i}.pt")