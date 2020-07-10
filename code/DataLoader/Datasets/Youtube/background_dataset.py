import glob
import random

import cv2
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from pathlib import Path
import os
from collections import defaultdict
import json
from torch.utils import data


class Backgrounds(data.Dataset):

    def __init__(self, start_index=torch.tensor([0])):
        self.file_path = Path("data/Images/other/background_dataset")
        self.start_index = start_index[0].item()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        img = Image.open(self.file_path / "01.jpg")
        img = img.resize((512, 257))
        to_tensor = T.ToTensor()
        return idx, (to_tensor(img).to(self.device), to_tensor(img).to(self.device))

    def show(self):
        img = Image.open(self.file_path / "01.jpg")
        img = img.resize((512, 257))
        img.show()

    def set_start_index(self, idx):
        self.start_index = idx[0].item()

if __name__ == "__main__":
    dataset = Backgrounds()
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    dataset.show()

    for i, img in enumerate(loader):
        print(img.shape)
        break
