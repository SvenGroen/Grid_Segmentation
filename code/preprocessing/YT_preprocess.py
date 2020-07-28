import glob
import json
import random
import time

import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
import os
from collections import defaultdict
from pathlib import Path

from sklearn.model_selection import train_test_split

seed = 12345
random.seed(seed)
np.random.seed(seed)


def add_noise(image):
    row, col, ch = image.shape
    mean = 0
    # var = 0.1
    # sigma = var**0.5
    gauss = np.random.normal(mean, 1.5, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = image + gauss
    return noisy


vid_path = Path(Path.cwd()) / "data/Videos/YT_originals_sorted_test"
video_names = [vid.stem for vid in vid_path.glob("*")]
output_size = (int(2048 / 4), int(1080 / 4))
fps = 29
lower_green = np.array([0, 150, 0])
upper_green = np.array([150, 255, 150])
MAX_DURATION = 4

splits = ["train", "test"]


fourcc = cv2.VideoWriter_fourcc(*"mp4v")
for split in splits:
    out_path = Path("data/Images/Greenscreen_Video_frames") / split
    video_names = [vid.stem for vid in (vid_path/split).glob("*")]

    label_out_path = out_path / "labels"
    input_out_path = out_path / "Input"
    label_out_path.mkdir(parents=True, exist_ok=True)
    input_out_path.mkdir(parents=True, exist_ok=True)
    bgpath = Path("data/Images/Backgrounds2") / split
    frame_counter = 0
    out_log = defaultdict(list)
    for i, vid in enumerate(video_names):
        print("--------------------------------")
        print("video: ", vid)
        new_vid_marker = True
        cap = cv2.VideoCapture(str(vid_path / split / vid) + ".mp4")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
        cap.set(cv2.CAP_PROP_FPS, fps)
        print("old frame rate: {}; new frame rate: {}".format(frame_rate, fps))
        print("max frames: ", total_frames)
        bgimg = [img for img in bgpath.glob("*")]
        bgimg = str(bgimg[i % len(bgimg)])
        bgimg = cv2.imread(bgimg)
        bgimg = cv2.resize(bgimg, output_size)
        bgimg = np.clip(add_noise(bgimg), a_min=0, a_max=255)
        start = True

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, output_size)
                mask = cv2.inRange(frame, lower_green,
                                   upper_green)  # create a mask for the label and the background swap
                mask = np.expand_dims(mask, axis=-1)
                label = np.where(mask, (0, 0, 0), (255, 255, 255))
                out_img = np.where(mask, add_noise(bgimg), frame)
                out_name = str(frame_counter).zfill(5) + ".jpg"
                cv2.imwrite(str(input_out_path / out_name), np.uint8(out_img))
                cv2.imwrite(str(label_out_path / out_name), np.uint8(label))
                out_log["Inputs"].append((str(input_out_path / out_name), int(new_vid_marker)))
                out_log["Labels"].append((str(label_out_path / out_name), int(new_vid_marker)))
                frame_counter += 1
                new_vid_marker = False
            else:
                break
        cap.release()
    with open(str(out_path / "out_log.json"), "w") as js:
        json.dump(dict(out_log), js)