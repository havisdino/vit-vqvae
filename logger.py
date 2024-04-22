from abc import ABC, abstractmethod

from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
import os


class Logger(ABC):
    @abstractmethod
    def log(self, info: dict, step: int):
        pass

    @abstractmethod
    def add_image(self, tag, images: Tensor, step: int):
        pass

    @abstractmethod
    def close(self):
        pass


class TensorBoardLogger(Logger):
    def __init__(self):
        super().__init__()
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        self.writer = SummaryWriter('logs')

    def log(self, info, step):
        for key, value in info.items():
            self.writer.add_scalar(key, value, step)
            
    def add_image(self, tag, images, step):
        self.writer.add_image(tag, images, step)

    def close(self):
        self.writer.close()
