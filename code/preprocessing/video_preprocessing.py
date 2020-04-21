import glob
import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
import os
from collections import defaultdict
from pathlib import Path

vid_path = Path(Path.cwd()) / "data/Videos/Examples"
backgroundname = Path("no_bg")
for vid in vid_path.glob("*.mp4"):
    cap = cv2.VideoCapture(str(vid))
    name = Path(vid.stem)
    frame_count = 0
    label_path = (Path.cwd() / Path("data/Images/Examples")/ backgroundname / name / Path("labels/"))
    frame_path = (Path.cwd() / Path("data/Images/Examples")/ backgroundname / name / Path("Input/"))
    label_path.mkdir(parents=True, exist_ok=True)
    frame_path.mkdir(parents=True, exist_ok=True)
    print("Video: ", name)
    while cap.isOpened():
        ret, frame = cap.read()
        fname = str(name) +"_"+ str(frame_count) + ".png"
        if ret:
            label = (np.any(frame, axis=2, keepdims=False) * 255.)
            cv2.imwrite(str(frame_path) + "\\"+ fname, frame)
            cv2.imwrite(str(label_path) + "\\"+ fname , label)
            frame_count += 1
        else:
            break
    cap.release()
print("All Videos processed")
