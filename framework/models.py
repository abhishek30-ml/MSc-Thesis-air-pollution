import torch
import torch.nn as nn

from timm.models.vision_transformer import Block, PatchEmbed
from diffusers import UNet2DModel, UNet2DConditionModel

from utils.position_emb import get_2d_sincos_pos_embed
from utils.patches import CustomPatchEmbed, unpatchify


class MaskedAutoencoderViT(nn.Module):
    """
    
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        dec_chans = (32,64,128,256),
        mlp_ratio=4.0,
        concat_voronoi=True,
        sensor_pos=None,
        unet_config = None
    ):
        super().__init__()

        self.patch_size = patch_size
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.in_chans = in_chans
        self.patch_embed = CustomPatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, embed_dim), requires_grad=False
        )
        self.concat_voronoi = concat_voronoi
        self.sensor_pos = sensor_pos

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=nn.LayerNorm,
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        # 
        self.decoder_embed = nn.Linear(embed_dim, patch_size[0]*patch_size[1]*4, bias=True)

        # Unet Decoder
        self.decoder_unet = UNet2DModel(in_channels=8,
                                        out_channels=4,
                                        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
                                        up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
                                        block_out_channels=dec_chans,
                                        attention_head_dim= 4
                                        )
        
        self.decoder_unet = UNet2DModel(**unet_config)
        

        # Positional embeddings
        pos_embed = get_2d_sincos_pos_embed(
            embed_dim=self.pos_embed.shape[-1],
            grid_size=self.patch_embed.grid_size,
            cls_token=False,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
            

    def unpatchify(self, x):
        """
        x: (N, L, D)
        imgs: (N, C, H, W)
        """
        if isinstance(self.patch_size ,int):
            patch_size = (self.patch_size, self.patch_size)
        else:
            patch_size = self.patch_size

        if isinstance(self.img_size ,int):
            image_size = (self.img_size, self.img_size)
        else:
            image_size = self.img_size
        
        return unpatchify(x, patch_size, image_size, number_of_channels=self.in_chans, overlap=True)
    
    def add_sensor_mask(self,x):
        N, C, H, W = x.shape  # channel, length, dim
        mask = torch.zeros_like(x).to(x.device)
        for i in range(C):
            for a,b in self.sensor_pos[i]:
                mask[:,i,b,a] = 1
        masked_batch = x*mask
        masked_batch = torch.cat((x, mask), dim=1)
        return masked_batch
    
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


    def forward_encoder(self, x):
        """
        Forward function for the encoding part.
        """
        x = self.patch_embed(x)
        x = x + self.pos_embed

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    def forward_decoder(self, x, imgs):
        """
        Forward function for the decoding part.
        """
        # embed tokens
        x = self.decoder_embed(x)

        # Unpatchify
        x = self.unpatchify(x)

        # Unet Decoder
        if self.concat_voronoi:
            x1 = torch.cat((x, imgs), dim=1)
        else:
            x1 = self.add_sensor_mask(x)

        x1, pad_h, pad_w = self._pad_input(x1)
        x_decode = self.decoder_unet(x1, 0, return_dict=False)[0]
        x_decode = self._unpad_output(x_decode, pad_h, pad_w)

        return x_decode, x

    

    def forward(self, imgs):
        """
        
        """
        x = self.forward_encoder(imgs)
        predicted_patches, encode_output = self.forward_decoder(x, imgs)

        return predicted_patches, encode_output




class ConditionEncoder(nn.Module):

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, embed_dim), requires_grad=False
        )
        
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=nn.LayerNorm,
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        
        # Positional embeddings
        pos_embed = get_2d_sincos_pos_embed(
            embed_dim=self.pos_embed.shape[-1],
            grid_size=self.patch_embed.grid_size,
            cls_token=False,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
            
    
    def forward(self, x):
        """
        Forward function for the encoding part.
        """
        x = self.patch_embed(x)
        x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x


def Unet2D(config):

    return UNet2DModel(**config)

def Unet2DCondition(config):

    return UNet2DConditionModel(**config)