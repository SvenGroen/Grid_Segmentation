import glob

import cv2
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch
import numpy as np



class VideoDataset(Dataset):
    def __init__(self, filename):
        self.filename = str(filename)
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.cap = cv2.VideoCapture("./videos/Outputs/" + self.filename + ".mp4")
        try:
            self.frames = np.moveaxis(np.load("./data/" + self.filename + ".npz")["frames"], -1, 1)
            self.labels = np.load("./data/" + self.filename + ".npz")["labels"]
            self.masks = np.moveaxis(np.load("./data/" + self.filename + ".npz")["masks"], -1, 1)
            print(
                "Frames of shape {} loaded successfully.\nLabels of shape {} loaded successfully.\nMasks of shape {} loaded successfully.\n".format(
                    self.frames.shape, self.labels.shape, self.masks.shape))
            print("Returned Dataloader Images will have shape: Batchsize + {},"
                  "\nReturned Dataloader Labels will have shape: Batchsize + {},".format(self.frames[0].shape,
                                                                                         self.labels[0].shape))


        except IOError:
            print("Could not load the files")
    def get_all_frames(self):
        return self.frames


    def change_background_pictures(self,number_of_rand_backgrounds=None):
        # Load Background images
        try:
            backgrounds = np.load("./images/Coco_val_2017/coco_raw.npz", allow_pickle=True)["data"]
        except IOError:
            backgrounds = np.array([cv2.imread(file) for file in
                                    glob.glob("./images/Coco_val_2017/*.jpg")])
            np.savez("./images/Coco_val_2017/coco_raw.npz", data=backgrounds)

        # only take part of the dataset if wnated
        if number_of_rand_backgrounds is None:
            number_of_rand_backgrounds = len(backgrounds)
        backgrounds = backgrounds[:number_of_rand_backgrounds]
        frames_new = []

        for i, frame in enumerate(self.frames):
            rnd_idx = np.random.choice(range(len(backgrounds)))
            background = backgrounds[rnd_idx]
            if background.shape != (self.frames.shape):
                background = np.moveaxis(cv2.resize(background, (2048, 1080)), -1, 0)
            else:
                print("Error occured trying to match the dimensions")
            frame = np.where((self.masks[i]), frame, background)
            frames_new.append(frame)
        self.frames = frames_new
        return None

    def change_background_videos(self,video_path):
        # Load Background images
        cap = cv2.VideoCapture(video_path)


        frames_new = []

        for i, frame in enumerate(self.frames):
            rnd_idx = np.random.choice(range(len(backgrounds)))
            background = backgrounds[rnd_idx]
            if background.shape != (self.frames.shape):
                background = np.moveaxis(cv2.resize(background, (2048, 1080)), -1, 0)
            else:
                print("Error occured trying to match the dimensions")
            frame = np.where((self.masks[i]), frame, background)
            frames_new.append(frame)
        self.frames = frames_new
        return None

    def show(self, what="raw"):
        if what == "raw":
            frames = np.moveaxis(self.frames, 1, -1)
        elif what == "labels":
            frames = self.labels.squeeze() * 1.
        for frame in frames:
            cv2.imshow(self.filename, frame)
            cv2.waitKey(33)
        cv2.destroyAllWindows()

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        return self.frames[idx].astype(np.uint8), self.labels[idx]# normalize
