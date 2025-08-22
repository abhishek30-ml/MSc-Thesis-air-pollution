import math
import torch.nn as nn

def patchify(imgs, patch_size, overlap=True):
        N,C,H,W = imgs.shape
        patch_h, patch_w = patch_size

        if H%patch_h != 0 or W%patch_w != 0:
                raise Exception("Image not divisible by patch size")
        
        # unfold helps in convolution style unfolding of the data. With stride=patch_size, we get individual patches of images
        if overlap:
                stride = (math.ceil(patch_h/2), math.ceil(patch_w/2))
        else:
                stride = patch_size
        imgs = nn.functional.unfold(imgs, kernel_size=patch_size, stride=stride)
        imgs = imgs.permute(0,2,1)

        return imgs


def unpatchify(patches, patch_size, image_size, number_of_channels=1, overlap=True):
    ###
    ### Takes a batch of patches (N, L, D) where D = patch_size**2*C
    ### and returns the batch of images (N, C, H, W)

    patches = patches.permute(0, 2, 1)
    # fold performes the opposite operation of unfold (creating images from convolution style patches)
    if overlap:
        stride = (math.ceil(patch_size[0]/2), math.ceil(patch_size[1]/2))
    else:
        stride = patch_size
    patches = nn.functional.fold(patches, output_size=image_size, kernel_size=patch_size, stride=stride)

    return patches
    

class CustomPatchEmbed(nn.Module):

    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super(CustomPatchEmbed, self).__init__()

        self.img_size = img_size
        self.patch_size = patch_size

        self.stride = (math.ceil(patch_size[0]/2), math.ceil(patch_size[1]/2))

        self.grid_size, self.num_patches = self._init_img_size()

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=self.stride)
        self.norm = nn.BatchNorm2d(embed_dim)

    def _init_img_size(self):
        grid_size_h = math.floor((self.img_size[0] - self.patch_size[0])/self.stride[0]) +1 
        grid_size_w = math.floor((self.img_size[1] - self.patch_size[1])/self.stride[1]) +1 

        grid_size = (int(grid_size_h), int(grid_size_w))
        num_patches = grid_size[0]*grid_size[1]

        return grid_size, num_patches
    
    def forward(self, x):
        N,C,W,H = x.shape
        x = self.proj(x)
        #x = self.norm(x)
        x = x.flatten(2).transpose(1,2)
        
        return x