from argparse import ArgumentParser
import torch

from logger import TensorBoardLogger
from trainer import Trainer
from utils import create_data_loader, init_weights, new_modules_from_checkpoint, new_modules_from_config, vae_summary

import config as C


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--traindata', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--from-checkpoint', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--grad-accum-interval', type=int, default=1)
    parser.add_argument('--checkpoint-interval', type=int, default=10)
    parser.add_argument('--checkpoint-retention', type=int, default=5)
    parser.add_argument('--use-amp', type=bool, default=True)
    parser.add_argument('--n-steps', type=int, default=10000)

    args = parser.parse_args()

    if args.from_checkpoint is not None:
        checkpoint = torch.load(args.from_checkpoint, args.device)
        model, optimizer, lr_scheduler, grad_scaler = new_modules_from_checkpoint(checkpoint)
        init_step = checkpoint['global_step']
    else:
        model, optimizer, lr_scheduler, grad_scaler = new_modules_from_config()
        model.apply(init_weights)
        init_step = 0
    
    model.to(args.device)
    vae_summary(model)
    
    trainloader = create_data_loader(C.img_size, args.batch_size, args.traindata)
    trainer = Trainer(
        model=model, optimizer=optimizer, lr_scheduler=lr_scheduler, grad_scaler=grad_scaler,
        grad_accum_interval=args.grad_accum_interval, device=args.device, global_step=init_step,
        use_amp=args.use_amp, checkpoint_interval=args.checkpoint_interval,
        checkpoint_retention=args.checkpoint_retention
    )

    logger = TensorBoardLogger()
    trainer.fit(trainloader, args.n_steps, logger)
