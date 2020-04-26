import glob
import cv2
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from pathlib import Path
import os
from collections import defaultdict
import json
from torch.utils import data


class Example_NY(data.Dataset):

    input_path = Path("data/Images/Examples/NY-city")
    label_path = Path("data/Images/Examples/raw")

    def __init__(self, augmentation_transform:list=None, norm_ImageNet=False, name:str="Grandmother"):
        transform = []
        if isinstance(augmentation_transform, list) and augmentation_transform is not None:
            transform = transform + augmentation_transform

        transform.append(transforms.ToTensor())
        transform_lbl = transform
        if norm_ImageNet:
            transform.append(transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]))
            transform_lbl = transform[:-1]
        self.in_path = Example_NY.input_path / Path(name + "/Input/")
        self.lbl_path = Example_NY.label_path / Path(name + "/labels/")
        self.transform = transforms.Compose(transform)
        self.transform_lbl = transforms.Compose(transform_lbl)
        self.dataset = defaultdict(list)
        try:
            with open("code/DataLoader/Datasets/Examples/NY/NY.json", "r") as js:
                self.dataset = json.load(js)
        except FileNotFoundError or IOError or FileExistsError:
            self.dataset = self.create_json()

    def create_json(self):
        dset = defaultdict(list)

        for img in self.in_path.glob("*.jpg"):
            dset["Inputs"].append(str(Path(img)))
        for lbl in self.lbl_path.glob("*.jpg"):
            dset["Labels"].append(str(Path(lbl)))
        with open("code/DataLoader/Datasets/Examples/NY/NY.json", "w") as js:
            json.dump(dict(dset), js)
        return dset

    # def load_frames(self):
    #     self.frames=[]
    #     self.labels=[]

    #     # lbl_files = sorted(lbl_files, key=lambda x: str(os.path.splitext(x)[0].split("_")[-1])) # sorts the files based on their last number
    #     # inp_files = sorted(inp_files, key=lambda x: str(os.path.splitext(x)[0].split("_")[-1])) # sorts the files based on their last number

    #     for img in self.in_path.glob("*.jpg"):
    #         self.frames.append(self.transform(Image.open(img)))
    #     for lbl in self.lbl_path.glob("*.jpg"):
    #         self.labels.append(self.transform(Image.open(lbl)))
    #     return self.frames, self.labels

    def __len__(self):
        return len(self.dataset["Inputs"])

    def __getitem__(self, idx):
        img = Image.open(str(Path.cwd() / Path(self.dataset["Inputs"][idx])))
        inp = self.transform(img)
        lbl = self.transform_lbl(Image.open(str(Path.cwd() / Path(self.dataset["Labels"][idx]))))
        if torch.cuda.is_available():
            return inp.cuda(), lbl.squeeze(0).cuda()
        else:
            return inp, lbl.squeeze(0)
        # return inp, lbl.squeeze(0)

    # def show(self, what="raw"):
    #     if what == "raw":
    #         print("Showing Input Frames as Video. This may take some time, since the images need to be converted.")
    #         frames = [cv2.cvtColor(np.moveaxis(f.numpy(),0,-1),cv2.COLOR_RGB2BGR) for f in self.frames]
    #         for frame in frames:            
    #             cv2.imshow("Show()", frame)
    #             cv2.waitKey(30)
    #         cv2.destroyAllWindows()
    #     elif what == "labels":
    #         frames = [lbl.squeeze().numpy() * 1. for lbl in self.labels]
    #         # print(frames.shape)
    #         for frame in frames:       
    #             cv2.imshow("Show()", frame)
    #             cv2.waitKey(30)
    #         cv2.destroyAllWindows()


if __name__ == "__main__":

    dataset = Example_NY(name="a")
    loader = DataLoader(dataset=dataset, batch_size=1)
    inp, label = next(iter(loader))
    # dataset.show(what="raw")
