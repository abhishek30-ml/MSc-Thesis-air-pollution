import numpy as np
import yaml
import torch
from torch.utils.data import DataLoader

from diffusers.schedulers import EDMDPMSolverMultistepScheduler as scheduler
from diffusers.utils.torch_utils import randn_tensor


from framework.models import Unet2DCondition, ConditionEncoder
from dataloader.simulation import load_data
from utils.noise_scheduler import Karras_sigmas_lognormal

class Cond_Diffusion_Model():
    def __init__(self, img_size, unet_config, cond_config, lrate=0.001, weight_decay=0, eta_min=0.0001, t_max=100, lr_schedule=True, load_checkpoint=False, device='cuda', in_channels=4):
        super(Cond_Diffusion_Model, self).__init__()

        self.model_path = 'model_weights/diffusion.pth'
        self.lr_schedule = lr_schedule
        self.train_losses_list = []
        self.val_losses_list = []
        self.in_chanl = in_channels

        self.cond_model = ConditionEncoder(img_size=img_size,      
                               patch_size=cond_config['patch_size'],
                                in_chans=cond_config['in_chans'],
                                embed_dim=cond_config['embed_dim'],
                                depth=cond_config['depth'],
                                num_heads=cond_config['num_heads'],
                                mlp_ratio=cond_config['mlp_ratio']
                            )
        
        self.model = Unet2DCondition(unet_config)

        self.cond_model.to(device)
        self.model.to(device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=lrate, weight_decay=weight_decay)
        if lr_schedule:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=t_max, eta_min=eta_min)

        self.lastepoch = 0
        torch.serialization.add_safe_globals([np._core.multiarray._reconstruct])
        if load_checkpoint==True:
            print("Loading existing model from : ", self.model_path)
            checkpoint = torch.load(self.model_path, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.cond_model.load_state_dict(checkpoint['cond_model_state_dict'])
            self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
            if lr_schedule:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.lastepoch = checkpoint['epoch']
            self.train_losses_list = checkpoint['train_loss']
            self.val_losses_list = checkpoint['val_loss']

        
    def save_model(self, i):
        if self.lr_schedule:
            state = {
                'epoch': i,
                'model_state_dict': self.model.state_dict(),
                'cond_model_state_dict': self.cond_model.state_dict(),
                'optimizer_state_dict': self.optim.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'train_loss': self.train_losses_list,
                'val_loss': self.val_losses_list
            }
        else:
            state = {
                    'epoch': i,
                    'model_state_dict': self.model.state_dict(),
                    'cond_model_state_dict': self.cond_model.state_dict(),
                    'optimizer_state_dict': self.optim.state_dict(),
                    'train_loss': self.train_losses_list,
                    'val_loss': self.val_losses_list
                }

        torch.save(state, self.model_path)

    def best_model(self):
        checkpoint = torch.load(self.model_path, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.cond_model.load_state_dict(checkpoint['cond_model_state_dict'])

    
    def _pad_input(self, x, multiple=16):
        """
        Pad input tensor (B, C, H, W) so H and W are divisible by `multiple`.
        Returns padded tensor and amount of padding applied.
        """
        B, C, H, W = x.shape
        pad_h = (multiple - H % multiple) % multiple
        pad_w = (multiple - W % multiple) % multiple

        # Pad right and bottom only (left/top = 0)
        padded = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h))  # pad = (left, right, top, bottom)

        return padded, pad_h, pad_w
    
    def _unpad_output(self, output, pad_h, pad_w):
        """
        Remove padding from model output (B, C, H, W)
        """
        if pad_h > 0:
            output = output[:, :, :-pad_h, :]
        if pad_w > 0:
            output = output[:, :, :, :-pad_w]
        return output

    
    def back_sampling(self, img_size, mask, vt, noise_scheduler, org_img, generator, num_inf_steps, device):

        image = randn_tensor(img_size, generator=generator, device=device, dtype=self.model.dtype)
        noise = image.clone()

        noise_scheduler.set_timesteps(num_inf_steps)
        vt_cond = self.cond_model(vt)

        for i,t in enumerate(noise_scheduler.timesteps):
            x_in = noise_scheduler.scale_model_input(image, t)

            x_in, pad_h, pad_w = self._pad_input(x_in)
            vt_pad, _, _ = self._pad_input(vt)

            if self.in_chanl==8:
                x_in_concat = torch.cat((x_in, vt_pad), dim=1)
            else:
                x_in_concat = x_in

            model_output = self.model(x_in_concat, t, encoder_hidden_states=vt_cond, return_dict=False)[0]
            model_output = self._unpad_output(model_output, pad_h, pad_w)
            # predict Ftheta
            model_output = model_output * (1-mask) + org_img*(mask) 
            
            # Dtheta (precondition output is performed within step function)
            image = noise_scheduler.step(model_output, t, image, return_dict=False)[0]

            tmp_known_points = org_img.clone()
            if i < len(noise_scheduler.timesteps) - 1:
                noise_timestep = noise_scheduler.timesteps[i+1]
                tmp_known_points = noise_scheduler.add_noise(tmp_known_points, noise, torch.tensor([noise_timestep]))

            image = image * (1-mask) + tmp_known_points* mask

        
        return image
    
    def get_sigmas(self, noise_scheduler, timesteps, n_dim=4, dtype=torch.float32, device='cuda'):
        # modified from diffusers/examples/dreambooth/train_dreambooth_lora_sdxl.py
        sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = noise_scheduler.timesteps.to(device)
        timesteps = timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
        
    
    def loss_fn(self, output_img, org_img, sigmas):

        alpha = (sigmas ** 2 + 0.5** 2) / (sigmas * 0.5) ** 2 
        final_loss = torch.mean((alpha * (output_img-org_img)**2))

        return final_loss
    
    def reconst_error(self, org_img, output_img):
        mse_loss = torch.nn.functional.mse_loss(org_img, output_img, reduction='mean')
        return mse_loss.item()

    
    def train_model(self, train_dataloader, val_dataloader, noise_sampler, noise_scheduler, num_epochs=40, device='cuda'):
        patience_counter=0
        for epoch in range(self.lastepoch, num_epochs):
            
            self.model.train()
            total_loss = 0.0
            for org_img, mask, vt in train_dataloader:

                # Condition with concat of Voronoi tesselation

                org_img = org_img.to(device)
                mask = mask.to(device)
                vt = vt.to(device)

                vt_cond = self.cond_model(vt)

                noise = torch.randn(org_img.shape,  device=device)
                batch_size = org_img.shape[0]

                indices = noise_sampler(batch_size, device='cpu')
                timesteps = noise_scheduler.timesteps[indices].to(device=device)
                #sigmas = noise_scheduler.sigmas[indices].to(device=device)

                noise = noise*(1-mask)
                noisy_images = noise_scheduler.add_noise(org_img, noise, timesteps)

                sigmas = self.get_sigmas(noise_scheduler, timesteps, len(noisy_images.shape), noisy_images.dtype)
                x_in = noise_scheduler.precondition_inputs(noisy_images, sigmas)

                x_in, pad_h, pad_w = self._pad_input(x_in)
                vt_pad, _, _ = self._pad_input(vt)

                if self.in_chanl==8:
                    x_in_concat = torch.cat((x_in, vt_pad), dim=1)
                else:
                    x_in_concat = x_in

                self.optim.zero_grad()
                model_output = self.model(x_in_concat, timesteps, encoder_hidden_states=vt_cond, return_dict=False)[0]

                model_output = self._unpad_output(model_output, pad_h, pad_w)

                model_output = noise_scheduler.precondition_outputs(noisy_images, model_output, sigmas)
                

                # Final output loss
                loss = self.loss_fn(model_output, org_img, sigmas)
                error = self.reconst_error(org_img, model_output)
                total_loss += error

                loss.backward()
                self.optim.step()

            train_loss = total_loss/len(train_dataloader)
            self.train_losses_list.append(train_loss)

            if (epoch+1)%1==0:
                print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {(train_loss):.4f}")

            
            if (epoch+1)%20 ==0:
                val_loss,_ = self.evaluate_model(val_dataloader, epoch, device='cuda')
                self.val_losses_list.append(val_loss)
            
            if self.lr_schedule:
                self.scheduler.step(val_loss)

            #self.save_model(epoch)

            # Early stop
            
            if epoch>5:
                if train_loss < min(self.train_losses_list[:-1]):
                    self.save_model(epoch)
                    patience_counter = 0
                else:
                    patience_counter+=1
                    if patience_counter>12:
                        print('Patience Counter reached')
                        return self.train_losses_list, self.val_losses_list

        return self.train_losses_list, self.val_losses_list
    
    def evaluate_model(self, dataloader, epoch, device='cuda'):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            noise_scheduler_eval = scheduler(algorithm_type='sde-dpmsolver++')

            for mask, vt, org_img in dataloader:

                # Condition with concat of Voronoi tesselation

                org_img = org_img.to(device)
                mask = mask.to(device)
                vt = vt.to(device)
                generator = torch.Generator(device=device).manual_seed(42)

                sample_images = self.back_sampling(org_img.shape, mask, vt, noise_scheduler_eval, org_img, generator, num_inf_steps=50, device=org_img.device)
                #predicted_img, _ = self.model(masked_img)

                loss = self.reconst_error(sample_images, org_img)
                total_loss+=loss

            if (epoch+1)%1==0:
                print(f"Validation Loss: {(total_loss/len(dataloader)):.4f}")
                print('-------------------------------------')

            return total_loss/len(dataloader), sample_images
      
    def ensemble_prediction(self, org_img, mask, vt, num_inf_steps, num_ensem_steps, device):
        self.model.eval()
        ensem_output_list = []
        ensemble_output = torch.zeros_like(org_img)
        with torch.no_grad():
            for i in range(num_ensem_steps):
                noise_scheduler_ensem = scheduler(algorithm_type='sde-dpmsolver++')
                org_img = org_img.to(device)
                mask = mask.to(device)
                vt = vt.to(device)

                sample_img = self.back_sampling(org_img.shape, mask, vt, noise_scheduler_ensem, org_img, generator=None, num_inf_steps=num_inf_steps, device=org_img.device)
                ensemble_output += sample_img.cpu()
                ensem_output_list.append(ensemble_output/(i+1))

        return ensem_output_list
    
    def ensemble_inference_prediction(self, org_img, mask, vt, num_inf_steps, device):
        self.model.eval()
        ensem_output_list = []
        with torch.no_grad():
            org_img = org_img.to(device)
            mask = mask.to(device)
            vt = vt.to(device)

            for i in range(len(num_inf_steps)):
                noise_scheduler_ensem = scheduler(algorithm_type='sde-dpmsolver++')
                
                sample_img = self.back_sampling(org_img.shape, mask, vt, noise_scheduler_ensem, org_img, generator=None, num_inf_steps=num_inf_steps[i], device=org_img.device)
                ensem_output_list.append(sample_img.cpu())

        return ensem_output_list

def load_model(img_size, load_checkpoint):
    with open("config/train_model.yaml", "r") as f:
        config = yaml.safe_load(f)


    lr = config["learning_rate"]
    lr_schedule = config['lr_schedule']
    weight_decay = config['weight_decay']
    eta_min = config['eta_min']
    t_max = config['t_max']
    device = config['device']

    unet = config['model']['diffusion']['unet_model']
    cond_encoder = config['model']['diffusion']['condition_encoder']

    unet_config = {
        'sample_size': (80,112),
        'in_channels': unet['in_channels'],
        'out_channels': unet['out_channels'],
        'time_embedding_type': unet['time_embedding_type'],
        'flip_sin_to_cos': unet['flip_sin_to_cos'],
        'down_block_types': unet['down_block_types'],
        'up_block_types': unet['up_block_types'],
        'block_out_channels': unet['block_out_channels'],
        'act_fn': unet['act_fn'],
        'cross_attention_dim': unet['cross_attention_dim'],
        'attention_head_dim': unet['attention_head_dim']
    }

    cond_config = {
        'patch_size': cond_encoder['patch_size'],
        'in_chans': cond_encoder['in_chans'],
        'embed_dim': cond_encoder['embed_dim'],
        'depth': cond_encoder['depth'],
        'num_heads': cond_encoder['num_heads'],
        'mlp_ratio': cond_encoder['mlp_ratio']
    }

    base_model = Cond_Diffusion_Model(img_size, unet_config, cond_config, lrate=lr, weight_decay=weight_decay,
                                      eta_min=eta_min, t_max=t_max, lr_schedule=lr_schedule, load_checkpoint=load_checkpoint,
                                      device=device, in_channels=unet['in_channels'])
    
    return base_model
    

def train_model(load_checkpoint):

    with open("config/train_model.yaml", "r") as f:
        config = yaml.safe_load(f)

    train_idx = 7000
    img_size = (80,112)

    batch_size = config['batch_size']
    device = config['device']
    epochs = config['epochs']

    train_dataset_polair, val_dataset_polair = load_data(normalisation=True, train_idx=train_idx)
    train_dataloader = DataLoader(train_dataset_polair, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset_polair, batch_size=8, shuffle=False)

    noise_config = config['noise_scheduler']


    P_mean = noise_config['P_mean']
    P_std = noise_config['P_std']

    img,_,_ = next(iter(val_dataloader))
    img_size = img[0][0].shape

    base_model = load_model(img_size, load_checkpoint)
    
    noise_scheduler = scheduler(algorithm_type='sde-dpmsolver++')
    noise_sampler = Karras_sigmas_lognormal(noise_scheduler.sigmas, P_mean=P_mean, P_std=P_std)  #1.2, 1.7

    train_loss, val_loss = base_model.train_model(train_dataloader, val_dataloader, noise_sampler, noise_scheduler, num_epochs=epochs, device=device)

