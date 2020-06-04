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


class Youtube_Greenscreen_multiple(data.Dataset):
    with open("data/Images/Greenscreen_Video_frames/out_log.json", "r") as json_file:
        data = json.load(json_file)

    def __init__(self, transforms: list = None, norm_ImageNet=False, num_return=3):
        # set data
        self.data = Youtube_Greenscreen.data
        self.transform, self.transform_out = self.preprocess_transforms(norm_ImageNet=norm_ImageNet,
                                                                        transforms=transforms)
        self.num_return=num_return


    def __len__(self):
        return len(self.data["Inputs"])

    def __getitem__(self, idx):
        tmp = None
        for i in range(self.num_return):
            img = Image.open(str(Path.cwd() / Path(self.data["Inputs"][idx])))
            lbl = Image.open(str(Path.cwd() / Path(self.data["labels"][idx])))
            state = random.getstate()  # makes sure the transformations are applied equally
            inp = self.transform_out(self.transform(img)).unsqueeze(0)
            random.setstate(state)
            lbl = self.transform(lbl).unsqueeze(0)
            if tmp is not None:
                inp = torch.cat((inp, tmp[0]), 0)
                lbl = torch.cat((lbl, tmp[1]), 0)

            tmp = (inp, lbl)


        if torch.cuda.is_available():
            return inp.cuda(), lbl.cuda()
        else:
            return inp, lbl

    def show(self, num_images, start_idx: int = 0, random_images=True):

        out = []
        to_PIL = T.ToPILImage()
        for i in range(start_idx, num_images):
            if random_images:
                indx = np.random.randint(0, len(self))
            else:
                indx = 0
            img = Image.open(self.data["Inputs"][indx])
            lbl = Image.open(self.data["labels"][indx])
            state = random.getstate()
            img = self.transform_out(self.transform(img))
            random.setstate(state)
            lbl = self.transform(lbl)
            out.append(hstack([to_PIL(img), to_PIL(lbl)]))
        result = vstack(out)
        result.show()

    def preprocess_transforms(self, norm_ImageNet, transforms):
        transform = []
        transform_out = [T.ToPILImage()]

        if isinstance(transforms, list) and transforms is not None:
            transform = transform + transforms

        if norm_ImageNet:
            transform_out += [T.ToTensor(), T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]), T.ToPILImage()]

        transform_blacklist = [type(T.ColorJitter()), type(T.Normalize(mean=0, std=0))]
        transform_out += [t for t in transform if type(t) in transform_blacklist]
        transform = [t for t in transform if type(t) not in transform_blacklist]

        transform.append(T.ToTensor())
        transform_out.append(T.ToTensor())
        return T.Compose(transform), T.Compose(transform_out)


# --------- Visualization functions -----------
def vstack(images):
    if len(images) == 0:
        raise ValueError("Need 0 or more images")

    if isinstance(images[0], np.ndarray):
        images = [Image.fromarray(img) for img in images]
    width = max([img.size[0] for img in images])
    height = sum([img.size[1] for img in images])
    stacked = Image.new(images[0].mode, (width, height))

    y_pos = 0
    for img in images:
        stacked.paste(img, (0, y_pos))
        y_pos += img.size[1]
    return stacked


def hstack(images):
    if len(images) == 0:
        raise ValueError("Need 0 or more images")

    if isinstance(images[0], np.ndarray):
        images = [Image.fromarray(img) for img in images]
    width = sum([img.size[0] for img in images])
    height = max([img.size[1] for img in images])
    stacked = Image.new(images[0].mode, (width, height))

    x_pos = 0
    for img in images:
        stacked.paste(img, (x_pos, 0))
        x_pos += img.size[0]
    return stacked


if __name__ == "__main__":
    transform = [T.RandomPerspective(distortion_scale=0.1), T.ColorJitter(0.5, 0.5, 0.5),
                 T.RandomAffine(degrees=10, scale=(1, 2)), T.RandomGrayscale(p=0.1), T.RandomGrayscale(p=0.1),
                 T.RandomHorizontalFlip(p=0.7)]

    dataset = Youtube_Greenscreen()
    loader = DataLoader(dataset=dataset, batch_size=4)
    to_pil = T.ToPILImage()
    inp, label = next(iter(loader))
    # for img in inp[0,:,:,:,:]:
    #     img=to_pil(img)
    #     img.show()

    dataset.show(10, random_images=False)
