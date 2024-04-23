from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from logger import Logger
from modules.vae import VAE
from utils import save_checkpoint


@dataclass
class Trainer:
    model: VAE
    optimizer: torch.optim.Optimizer
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler
    grad_scaler: torch.cuda.amp.GradScaler
    grad_accum_interval: int
    device: Any
    global_step: int = 0
    use_amp: bool = True
    max_grad_norm: float = 1.0
    checkpoint_interval: int = 10
    checkpoint_retention: int = 5

    def __post_init__(self):
        self.substep = 0

    def train_step(self, x, beta=0.25):
        self.model.train()
        with torch.autocast(self.device, torch.float16, self.use_amp):
            x_pred, ze, zq = self.model(x)
            self.reconstruction_loss = F.mse_loss(x_pred, x)
            self.codebook_loss = F.mse_loss(zq, ze.detach())
            self.commitment_loss = beta * F.mse_loss(ze, zq.detach())
            self.loss = self.reconstruction_loss + self.codebook_loss + self.commitment_loss
            self.loss /= self.grad_accum_interval

        self.grad_scaler.scale(self.loss).backward()
        self.substep += 1

    def accumulate_gradients(self):
        self.grad_scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        self.optimizer.zero_grad()
        self.lr_scheduler.step()

        self.global_step += 1

    def get_batch_loss(self):
        return self.loss.detach().item() * self.grad_accum_interval

    def info(self):
        return {
            'loss/train': self.get_batch_loss(),
            'reconstruction_loss/train': self.reconstruction_loss.detach().item(),
            'codebook_loss/train': self.codebook_loss.detach().item(),
            'commitment_loss/train': self.commitment_loss.detach().item()
        }

    def fit(self, train_loader, n_steps, logger: Logger):
        bar = tqdm(total=n_steps)
        n_steps += self.global_step
        data_iter = iter(train_loader)
        while self.global_step < n_steps:
            try:
                x = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                continue
            x = x.to(self.device)
            self.train_step(x)
            if self.substep % self.grad_accum_interval == 0:
                self.accumulate_gradients()
                logger.log(self.info(), self.global_step)

                if self.global_step % self.checkpoint_interval == 0:
                    save_checkpoint(
                        self.model, self.optimizer, self.lr_scheduler,
                        self.grad_scaler, self.global_step,
                        self.global_step - self.checkpoint_interval * self.checkpoint_retention
                    )
            bar.update()
