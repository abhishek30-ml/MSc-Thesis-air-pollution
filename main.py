
import argparse

from training.vit_unet import train_model as model_vitunet
from training.unet import train_model as model_unet
from training.diffusion import train_model as model_diffusion


def run(model, load_cp):
    if model == 'vit_unet':
        model_vitunet(load_cp)
    elif model == 'unet':
        model_unet(load_cp)
    elif model == 'diffusion':
        model_diffusion(load_cp)
    else:
        raise Exception('Model type not valid')
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='diffusion', type=str)
    parser.add_argument('--load_checkpoint', default=True, type=bool)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.model, args.load_checkpoint)