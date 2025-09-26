import pandas as pd
from PIL import Image
import os

from torch.utils.data import Dataset
from torchvision import transforms
class Cifar10Dataset(Dataset):
    """Custom Dataset for loading CelebA face images"""

    def __init__(self, data, transform=None):
    
        self.img_dir = img_dir
        self.csv_path = csv_path
        self.img_names = df.index.values
        self.y = df['Male'].values
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir,
                                      self.img_names[index]))
        
        if self.transform is not None:
            img = self.transform(img)
        
        label = self.y[index]
        return img, label

    def __len__(self):
        return self.y.shape[0]