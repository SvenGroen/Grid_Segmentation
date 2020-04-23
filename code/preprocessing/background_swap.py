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
all_vids = [vid for vid in vid_path.glob("*.mp4")]
processed_vids = all_vids[1:2]
print(processed_vids)
backgroundname = Path("NY-city")
backgroundimage = Path.cwd() / "data/Images/Examples/NY-city/NY-city.jpg"

for vid in vid_path.glob("*.mp4"):
    if vid in processed_vids: # only process certain videos
        cap = cv2.VideoCapture(str(vid))
        w = int(cap.get(3))
        h = int(cap.get(4))
        backgroundimage = cv2.resize(cv2.imread(str(backgroundimage)), (w,h))
        name = Path(vid.stem)
        frame_count = 0
        frame_path = (Path.cwd() / Path("data/Images/Examples")/ backgroundname / name / Path("Input/"))
        frame_path.mkdir(parents=True, exist_ok=True)
        print("Video: ", name)
        while cap.isOpened():
            ret, frame = cap.read()
            num_filler = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
            fname = str(name) +"_"+ str(frame_count).zfill(num_filler) + ".jpg"
            if ret:
                mask = np.where((frame==0), False, True) # get boolean mask for image
                frame = np.where((mask), frame, backgroundimage) # replace relevant pixels
                cv2.imwrite(str(frame_path / Path(fname)), frame)
                frame_count += 1
            else:
                break
        cap.release()
print("All Videos processed")
