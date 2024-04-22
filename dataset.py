import os
from torch.utils.data import Dataset
from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, directory, transform=None):
        super().__init__()
        
        self.transform = transform
        self.directory = directory
        self.fnames = [f for f in os.listdir(directory)
                       if f.endswith('.jpg') or f.endswith('.png')]
        
    def __len__(self):
        return len(self.fnames)
    
    def _path(self, idx):
        return os.path.join(self.directory, self.fnames[idx])
    
    def __getitem__(self, idx):
        path = self._path(idx)
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)
        return img
        