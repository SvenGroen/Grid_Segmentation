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

random.seed(42)
for split in ["train", "test"]:
    vid_path_inp = Path(Path.cwd()) / "data/Videos/rgb_dataset" / split / "Input"
    vid_path_lbl = Path(Path.cwd()) / "data/Videos/rgb_dataset" / split / "labels"
    video_names = [vid.stem for vid in vid_path_inp.glob("*")]
    output_size = (int(2048 / 4), int(1080 / 4))
    fps = 29

    out_path = Path("data/Images/other/rgb_dataset") / split
    label_out_path = out_path / "labels"
    input_out_path = out_path / "Input"
    label_out_path.mkdir(parents=True, exist_ok=True)
    input_out_path.mkdir(parents=True, exist_ok=True)
    count_inp = 0
    count_lbl = 0
    out_log = defaultdict(list)
    for vid in video_names:
        print("--------------------------------")
        print("video: ", vid)
        cap_inp = cv2.VideoCapture(str(vid_path_inp / vid) + ".mp4")
        while cap_inp.isOpened():
            ret, frame = cap_inp.read()
            out_name = str(input_out_path / (str(count_inp).zfill(5) + ".jpg"))
            if ret:
                cv2.imwrite(out_name, frame)
                out_log["Inputs"].append(out_name)
                count_inp += 1
            else:
                break
        cap_inp.release()
        frame = cv2.imread(str(vid_path_lbl / "label.jpg"))
        out_name = str(out_path /"labels/label.jpg")
        cv2.imwrite(out_name, frame)
        out_log["labels"].append(out_name)


    with open(str(out_path / "out_log.json"), "w") as js:
        json.dump(dict(out_log), js)