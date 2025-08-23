import numpy as np
import yaml
import torch
from torch.utils.data import DataLoader

from utils.patches import patchify
from framework.models import Unet2D
from dataloader.simulation import create_dataset

class Baseline_Model_Unet():
    def __init__(self, img_size, config=None, lrate=0.001, weight_decay=0, eta_min=0.0001, t_max=100, lr_schedule=True, load_checkpoint=False, device='cuda'):
        super(Baseline_Model_Unet, self).__init__()

        self.model_path = 'model_weights/unet.pth'
        self.img_size = img_size
        self.lr_schedule = lr_schedule
        self.train_losses_list = []
        self.val_losses_list = []
        self.chann_train_losses_list = []
        self.chann_val_losses_list = []
        self.device = device
        self.model = Unet2D(config)
        
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
    
    def call_model(self, vt):
        vt = vt.to(self.device)
        vt, pad_h, pad_w = self._pad_input(vt)
        self.model.eval()
        with torch.no_grad():
            predicted_img = self.model(vt, 0, return_dict=False)[0]
        predicted_img = self._unpad_output(predicted_img, pad_h, pad_w)
        return predicted_img

    
    def _train_model(self, train_dataloader, val_dataloader, num_epochs=40, device='cuda'):
        patience_counter=0
        for epoch in range(self.lastepoch, num_epochs):
            self.model.train()
            total_loss = 0.0
            total_chann_loss = np.array([0,0,0,0], dtype=np.float64)
            for org_img, _, vt in train_dataloader:

                org_img = org_img.to(device)
                vt = vt.to(device)

                self.optim.zero_grad()
                vt, pad_h, pad_w = self._pad_input(vt)
                predicted_img = self.model(vt, 0, return_dict=False)[0]
                predicted_img = self._unpad_output(predicted_img, pad_h, pad_w)

                # Final output loss
                loss, chann_loss = self.loss_fn(predicted_img, org_img)
                total_loss += loss.item()
                total_chann_loss += chann_loss.cpu().detach().numpy()

                final_loss = loss
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
    
    def _evaluate_model(self, dataloader, epoch, device='cuda'):
        self.model.eval()
        total_loss = 0.0
        total_chann_loss = np.array([0,0,0,0], dtype=np.float64)
        with torch.no_grad():
            for org_img, _, vt in dataloader:

                org_img = org_img.to(device)
                vt = vt.to(device)

                vt, pad_h, pad_w = self._pad_input(vt)
                predicted_img = self.model(vt, 0, return_dict=False)[0]
                predicted_img = self._unpad_output(predicted_img, pad_h, pad_w)

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

    model_conf = config['model']['Unet']

    unet_config = {
        'in_channels': model_conf['unet_in_channel'],
        'out_channels':model_conf['unet_out_channel'],
        'down_block_types': model_conf['down_block_types'],
        'up_block_types': model_conf['up_block_types'],
        'block_out_channels': model_conf['block_out_channels'],
        'attention_head_dim': model_conf['attention_head_dim']
    }

    base_model = Baseline_Model_Unet(img_size, config=unet_config, lrate=lr, weight_decay=weight_decay, eta_min=eta_min,
                                    t_max=t_max, lr_schedule=lr_schedule, load_checkpoint=load_checkpoint, device=device)
    
    return base_model
     


def train_model(load_checkpoint):

    with open("config/train_model.yaml", "r") as f:
        config = yaml.safe_load(f)

    train_idx = 7000

    batch_size = config['batch_size']
    device = config['device']
    epochs = config['epochs']

    train_dataset_polair, val_dataset_polair = create_dataset(normalisation=False, train_idx=train_idx)
    train_dataloader = DataLoader(train_dataset_polair, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset_polair, batch_size=8, shuffle=False)
    print('Dataset loading complete')

    img,_,_ = next(iter(val_dataloader))
    img_size = img[0][0].shape

    base_model = load_model(img_size, load_checkpoint)
    
    print('Training start')
    train_loss, val_loss, chann_train_loss, chann_val_loss = base_model._train_model(train_dataloader, val_dataloader, num_epochs=epochs, device=device)
