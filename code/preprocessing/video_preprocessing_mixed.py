import glob
import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
import os
from collections import defaultdict
from pathlib import Path

# This file extracts all frames from all videos in vid_path and saves them accordingly
# label frames are created based on the "blackness" of each pixel


vid_path = Path(Path.cwd()) / "data/Videos/Examples_Green"
backgroundname = Path("raw")

output_size = (int(2048 / 4),int(1080 / 4))
lower_green = np.array([0, 150, 0])
upper_green = np.array([100, 255, 120])

for vid in vid_path.glob("*.mp4"):
    cap = cv2.VideoCapture(str(vid))

    name = Path(vid.stem)
    frame_count = 0
    label_path = (Path.cwd() / Path("data/Images/Examples_Green") / backgroundname / name / Path("labels/"))
    frame_path = (Path.cwd() / Path("data/Images/Examples_Green") / backgroundname / name / Path("Input/"))
    label_path.mkdir(parents=True, exist_ok=True)
    frame_path.mkdir(parents=True, exist_ok=True)
    print("Video: ", name)
    while cap.isOpened():
        ret, frame = cap.read()

        num_filler = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
        fname = str(name) + "_" + str(frame_count).zfill(num_filler) + ".jpg"
        if ret:
            frame = cv2.resize(frame, output_size)
            label = cv2.inRange(frame, lower_green, upper_green)

            label = np.where(label, 0, 255)
            cv2.imwrite(str(frame_path) + "/" + fname, frame)
            cv2.imwrite(str(label_path) + "/" + fname, label)
            frame_count += 1
        else:
            break
    cap.release()

print("All Videos processed")
