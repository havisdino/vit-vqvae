from abc import ABC, abstractmethod

from torch import Tensor
from torch.utils.tensorboard import SummaryWriter


class Logger(ABC):
    @abstractmethod
    def log(self, info: dict, step: int):
        pass

    @abstractmethod
    def add_image(self, images: Tensor):
        pass

    @abstractmethod
    def close(self):
        pass


class TensorBoardLogger(Logger):
    def __init__(self):
        super().__init__()
        self.writer = SummaryWriter('logs')

    def log(self, info, step):
        for key, value in info.items():
            self.writer.add_scalar(key, value, step)

    def close(self):
        self.writer.close()
