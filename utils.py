import os
from time import time
from typing import Literal, overload
from torch import nn
import torch
import torch.utils
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import torch.utils.data
import math

import config as C
from dataset import ImageDataset
from modules.vae import VAE


def save_checkpoint(model, optimizer, lr_scheduler, grad_scaler, global_step, remove_checkpoint=None):
        if not os.path.exists('checkpoints'):
            os.makedirs('checkpoints')
            
        path = f'checkpoints/vqvae-{global_step}.pt'    
            
        last_kth = f'checkpoints/vqvae-{remove_checkpoint}.pt'
        
        if os.path.exists(last_kth):
            os.remove(last_kth)

        checkpoint = dict(
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            lr_scheduler=lr_scheduler.state_dict(),
            grad_scaler=grad_scaler.state_dict(),
            global_step=global_step
        )
        
        torch.save(checkpoint, path)


def init_weights(m):
    for p in m.parameters():
        nn.init.normal_(p, std=0.04)
        
        
def count_params(model):
    if isinstance(model, nn.DataParallel):
        n_params = sum(p.numel() for p in model.module.parameters())
    elif isinstance(model, nn.Module):
        n_params = sum(p.numel() for p in model.parameters())
    return n_params


def vae_summary(vae: VAE):
    decoder_nparams = count_params(vae.decoder)
    encoder_nparams = count_params(vae.encoder)
    print(f'Decoder parameters\t: {decoder_nparams:,}')
    print(f'Encoder parameters\t: {encoder_nparams:,}')
    print(f'Total parameters\t: {decoder_nparams + encoder_nparams:,}')
    
    
def unfold_to_patches(img, patch_size: list, strides: list):
    C = img.size(-3)
    assert len(patch_size) == 2 and len(strides) == 2
    assert img.ndim == 3 or img.ndim == 4
    batched = img.ndim == 4
    d1, d2 = patch_size
    s1, s2 = strides

    img = img.unfold(-1, d2, s2).unfold(-3, d1, s1)

    if not batched:
        img = img.permute(1, 2, 0, 4, 3).reshape(-1, C, d1, d2)
    else:
        B = img.size(0)
        img = img.permute(0, 2, 3, 1, 5, 4).reshape(B, -1, C, d1, d2)
    return img.flatten(-3)


def generate_file_name():
    return f'{int(time() * 1e9)}.png'


def save_images(imgs, dir, make_grid=True):
    if not os.path.exists(dir):
        os.makedirs(dir)
        
    if make_grid:
        fp = os.path.join(dir, generate_file_name())
        torchvision.utils.save_image(imgs / 255., fp)
    else:
        for img in imgs:
            fp = os.path.join(dir, generate_file_name())
            torchvision.utils.save_image(img / 255., fp)


def get_model_config():
    return dict(
        d_model=C.d_model,
        d_patch=C.d_patch,
        codebook_size=C.codebook_size,
        seqlen=C.seqlen,
        n_heads=C.n_heads,
        n_blocks=C.n_blocks,
        dff=C.dff,
        dropout=C.dropout
    )


def new_model_from_config():
    return VAE(**get_model_config())
        

def modify_config(config, **kwargs):
    for key, item in kwargs.items():
        setattr(config, key, item)


def new_model_from_checkpoint(checkpoint):
    config = checkpoint['config']
    modify_config(C, **config)
    print('Checkpoint loaded, default configurations might be ignored')
    return VAE(**config)


def lr_schedule(step):
    if step <= C.warmup_step:
        alpha = (C.peak_lr - C.init_lr) / (C.warmup_step ** 2)
        lr = alpha * (step ** 2) + C.init_lr
    else:
        beta = - C.warmup_step - C.down_weight * math.log(C.peak_lr - C.min_lr)
        lr = math.exp(-(step + beta) / C.down_weight) + C.min_lr
    return lr


def new_modules_from_checkpoint(checkpoint):
    model = new_model_from_checkpoint(checkpoint)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
    grad_scaler = torch.cuda.amp.GradScaler()

    optimizer.load_state_dict(checkpoint['optimizer'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    grad_scaler.load_state_dict(checkpoint['grad_scaler'])

    return model, optimizer, lr_scheduler, grad_scaler


def new_modules_from_config():
    model = new_model_from_config()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
    grad_scaler = torch.cuda.amp.GradScaler()

    return model, optimizer, lr_scheduler, grad_scaler


def create_data_loader(img_size, batch_size, directory) -> DataLoader:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(img_size),
        transforms.Lambda(lambda x: 2 * x - 1),
        transforms.Lambda(lambda x: unfold_to_patches(x, C.patch_size, C.strides))
    ])
    dataset = ImageDataset(directory, transform)
    return DataLoader(dataset, batch_size, shuffle=True, num_workers=2, prefetch_factor=2)
