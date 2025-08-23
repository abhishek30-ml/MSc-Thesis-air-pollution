import numpy as np
import yaml
import torch
from torch.utils.data import DataLoader

from utils.patches import patchify
from framework.models import MaskedAutoencoderViT
from dataloader.simulation import create_dataset

class ViTUnet_Model():
    def __init__(self, img_size, patch_size, embed_dim=32, depth=8, num_heads=8, unet_config=None, concat_voronoi=True, lrate=0.001, weight_decay=0, eta_min=0.0001, t_max=100, lr_schedule=True, load_checkpoint=False, device='cuda'):
        super(ViTUnet_Model, self).__init__()

        self.model_path = 'model_weights/vit.pth'
        self.img_size = img_size
        self.patch_size = patch_size
        self.lr_schedule = lr_schedule
        self.train_losses_list = []
        self.val_losses_list = []
        self.chann_train_losses_list = []
        self.chann_val_losses_list = []
        self.device = device
        self.model = MaskedAutoencoderViT(
            in_chans=4,
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            dec_chans = (32,64,128,256),
            mlp_ratio=4.0,
            concat_voronoi=concat_voronoi,
            unet_config=unet_config
        )
        
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
            self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
            if lr_schedule:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.lastepoch = checkpoint['epoch']
            self.train_losses_list = checkpoint['train_loss']
            self.val_losses_list = checkpoint['val_loss']
            self.chann_train_losses_list = checkpoint['chann_train_loss']
            self.chann_val_losses_list = checkpoint['chann_val_loss']
    
    def loss_fn(self, pred_patches, target_patches):
        mse_loss = torch.nn.functional.mse_loss(target_patches, pred_patches, reduction='none')
        chan_loss = torch.sum(mse_loss, dim=(0,2,3))
        final_loss = torch.nn.functional.mse_loss(target_patches, pred_patches, reduction='sum')

        return final_loss, chan_loss

    
    def patchify(self, imgs):
        """
        imgs: (N, C, H, W)
        x: (N, L, D)
        """
        if isinstance(self.patch_size ,int):
            patch_size = (self.patch_size, self.patch_size)
            return patchify(imgs, patch_size, overlap=False)
        else:
            return patchify(imgs, self.patch_size, overlap=False)
        
    def save_model(self, i):
        if self.lr_schedule:
            state = {
                'epoch': i,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optim.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'train_loss': self.train_losses_list,
                'val_loss': self.val_losses_list,
                'chann_train_loss': self.chann_train_losses_list,
                'chann_val_loss': self.chann_val_losses_list
            }
        else:
            state = {
                    'epoch': i,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optim.state_dict(),
                    'train_loss': self.train_losses_list,
                    'val_loss': self.val_losses_list,
                    'chann_train_loss': self.chann_train_losses_list,
                    'chann_val_loss': self.chann_val_losses_list
                }

        torch.save(state, self.model_path)

    def best_model(self):
        checkpoint = torch.load(self.model_path, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def call_model(self, vt):
        vt = vt.to(self.device)
        self.model.eval()
        with torch.no_grad():
            predicted_img,_ = self.model(vt)
        return predicted_img

    
    def _train_model(self, train_dataloader, val_dataloader, num_epochs=40, alpha=0.4, device='cuda'):
        patience_counter=0
        for epoch in range(self.lastepoch, num_epochs):
            self.model.train()
            total_loss = 0.0
            total_chann_loss = np.array([0,0,0,0], dtype=np.float64)
            for org_img, _, vt in train_dataloader:

                org_img = org_img.to(device)
                vt = vt.to(device)

                self.optim.zero_grad()
                predicted_img, encoder_img = self.model(vt)

                # Final output loss
                loss, chann_loss = self.loss_fn(predicted_img, org_img)
                total_loss += loss.item()
                total_chann_loss += chann_loss.cpu().detach().numpy()

                # Encoder Loss
                encoder_loss = torch.nn.functional.mse_loss(org_img, encoder_img, reduction='sum')

                final_loss = loss + alpha*encoder_loss
                final_loss.backward()
                self.optim.step()

            self.train_losses_list.append(total_loss/len(train_dataloader.dataset))
            self.chann_train_losses_list.append(total_chann_loss/len(train_dataloader.dataset))
            if (epoch+1)%1==0:
                print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {(total_loss/len(train_dataloader.dataset)):.4f}")

            val_loss, chann_loss = self._evaluate_model(val_dataloader,epoch, device='cuda')
            
            if self.lr_schedule:
                self.scheduler.step(val_loss)

            self.val_losses_list.append(val_loss)
            self.chann_val_losses_list.append(chann_loss)

            # Early stop
            if epoch>5:
                if val_loss < min(self.val_losses_list[:-1]):
                    self.save_model(epoch)
                    patience_counter = 0
                else:
                    patience_counter+=1
                    if patience_counter>12:
                        print('Patience Counter reached')
                        return self.train_losses_list, self.val_losses_list, self.chann_train_losses_list, self.chann_val_losses_list

        return self.train_losses_list, self.val_losses_list, self.chann_train_losses_list, self.chann_val_losses_list
    
    def _evaluate_model(self, dataloader,epoch, device='cuda'):
        self.model.eval()
        total_loss = 0.0
        total_chann_loss = np.array([0,0,0,0], dtype=np.float64)
        with torch.no_grad():
            for org_img, _, vt in dataloader:

                org_img = org_img.to(device)
                vt = vt.to(device)

                predicted_img, _ = self.model(vt)

                loss, chann_loss = self.loss_fn(predicted_img, org_img)
                total_loss+=loss.item()
                total_chann_loss+=chann_loss.cpu().detach().numpy()

            if (epoch+1)%1==0:
                print(f"Validation Loss: {(total_loss/len(dataloader.dataset)):.4f}")
                print('-------------------------------------')

            return total_loss/len(dataloader.dataset), total_chann_loss/len(dataloader.dataset)


def load_model(img_size, load_checkpoint):

    with open("config/train_model.yaml", "r") as f:
        config = yaml.safe_load(f)

    lr = config["learning_rate"]
    lr_schedule = config['lr_schedule']
    weight_decay = config['weight_decay']
    eta_min = config['eta_min']
    t_max = config['t_max']
    device = config['device']

    model_conf = config['model']['VitUnet']
    patch_size = model_conf['patch_size']
    embed_dim = model_conf['embed_dim']
    depth = model_conf['depth']
    num_heads = model_conf['num_heads']
    concat_voronoi = model_conf['concat_voronoi']

    unet_config = {
        'in_channels': model_conf['unet_in_channel'],
        'out_channels':model_conf['unet_out_channel'],
        'down_block_types': model_conf['down_block_types'],
        'up_block_types': model_conf['up_block_types'],
        'block_out_channels': model_conf['block_out_channels'],
        'attention_head_dim': model_conf['attention_head_dim']
    }

    base_model = ViTUnet_Model(img_size, patch_size, embed_dim=embed_dim, depth=depth, num_heads=num_heads, 
                                unet_config=unet_config, concat_voronoi=concat_voronoi, lrate=lr, weight_decay=weight_decay,
                                eta_min=eta_min, t_max=t_max, lr_schedule=lr_schedule, load_checkpoint=load_checkpoint, device=device)
    
    return base_model
     

def train_model(load_checkpoint):

    with open("config/train_model.yaml", "r") as f:
        config = yaml.safe_load(f)

    train_idx = 7000
    batch_size = config['batch_size']
    epochs = config['epochs']
    alpha = config['model']['VitUnet']['alpha']

    train_dataset_polair, val_dataset_polair = create_dataset(normalisation=False, train_idx=train_idx)
    train_dataloader = DataLoader(train_dataset_polair, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset_polair, batch_size=8, shuffle=False)
    print('Dataset loading complete')

    img,_,_ = next(iter(val_dataloader))
    img_size = img[0][0].shape

    base_model = load_model(img_size, load_checkpoint)
    
    print('Training start')
    train_loss, val_loss, chann_train_loss, chann_val_loss = base_model._train_model(train_dataloader, val_dataloader, num_epochs=epochs, alpha=alpha)

