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

random.seed(12345)
np.random.seed(12345)


def add_noise(image):
    row, col, ch = image.shape
    mean = 0
    # var = 0.1
    # sigma = var**0.5
    gauss = np.random.normal(mean, 1.0, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = image + gauss
    return noisy


bgpath = Path("data/Images/other/background_dataset")
vid_path = Path(Path.cwd()) / "data/Videos/rgb_dataset/raw"
video_names = [vid.stem for vid in vid_path.glob("*")]
output_size = (int(2048 / 4), int(1080 / 4))
fps = 29
lower_green = np.array([0, 150, 0])
upper_green = np.array([150, 255, 150])
MAX_DURATION = 4

splits = ["train", "test"]

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
for split in ["train", "test"]:
    out_path = Path("data/Videos/rgb_dataset") / split
    label_out_path = out_path / "labels"
    input_out_path = out_path / "Input"
    label_out_path.mkdir(parents=True, exist_ok=True)
    input_out_path.mkdir(parents=True, exist_ok=True)
    i = 0
    video_counter = 0

    for vid in video_names:
        print("--------------------------------")
        print("video: ", vid)
        cap = cv2.VideoCapture(str(vid_path / vid) + ".mp4")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
        cap.set(cv2.CAP_PROP_FPS, fps)
        print("old frame rate: {}; new frame rate: {}".format(frame_rate, fps))
        print("max frames: ", total_frames)
        bgimg = str([img for img in bgpath.glob("*")][0]) if split == "train" else \
            str([img for img in bgpath.glob("*")][1])
        bgimg = cv2.imread(bgimg)
        bgimg = cv2.resize(bgimg, output_size)
        cv2.imwrite(str(label_out_path / "label.jpg"), bgimg)
        start = True

        frame_counter = 0
        starter_frame = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                if start:
                    out_name = str(video_counter).zfill(5) + '.mp4'
                    out_input = cv2.VideoWriter(str(input_out_path / out_name),
                                                fourcc,
                                                fps, output_size)
                if starter_frame + MAX_DURATION * frame_rate == frame_counter:  # if 4 seconds passed break
                    video_counter += 1
                    starter_frame = frame_counter
                    if starter_frame + MAX_DURATION * frame_rate > total_frames:  # stop if last bit would not fit in 4 sec.
                        out_input.release()
                        break
                    out_name = str(video_counter).zfill(5) + '.mp4'
                    out_input = cv2.VideoWriter(str(input_out_path / out_name),
                                                fourcc,
                                                fps, output_size)


                start = False
                frame = cv2.resize(frame, output_size)
                mask = cv2.inRange(frame, lower_green,
                                   upper_green)  # create a mask for the label and the background swap
                mask = np.expand_dims(mask, axis=-1)

                out_img = np.where(mask, bgimg, frame)
                out_input.write(np.uint8(out_img))

                frame_counter += 1
                # cv2.imshow('frame', frame)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
            else:
                break
        cap.release()

        out_input.release()
