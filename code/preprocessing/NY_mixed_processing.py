import glob
import json
import random

import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
import os
from collections import defaultdict
from pathlib import Path


raw_frames_path = Path(Path.cwd()) / "data/Images/Examples_Green/raw"
video_names = [vid.stem for vid in raw_frames_path.glob("*")]

output_size = (int(2048), int(1080))
backgroundname = Path("NY-city_HD")
backgroundimage = cv2.imread(str(Path.cwd() / "data/Images/Examples/NY-city/NY-city.jpg"))
backgroundimage = cv2.resize(backgroundimage, output_size)

lower_green = np.array([0, 125, 0])
upper_green = np.array([100, 255, 120])
dataset_size = 1000
out_path =  Path("data/Images/Examples_Green") / backgroundname
label_out_path = out_path / "labels"
input_out_path = out_path / "Input"
label_out_path.mkdir(parents=True, exist_ok=True)
input_out_path.mkdir(parents=True, exist_ok=True)
out_log = defaultdict(list)
i = 0
for vid_name in video_names:
    inp_path = raw_frames_path / vid_name / "Input"
    for fname in os.listdir(inp_path):
        full_path = inp_path / fname
        rnd_img = cv2.imread(str(full_path))  # read in image
        out_img = cv2.resize(rnd_img, output_size)  # make sure dimensions are set
        mask = cv2.inRange(out_img, lower_green, upper_green) # create a mask for the label and the background swap
        mask = np.expand_dims(mask, axis=-1)
        label = np.where(mask, 0, 255)
        out_img = np.where(mask, backgroundimage, out_img)
        out_name = str(i).zfill(5) + ".jpg"
        out_log["labels"].append(str(label_out_path / out_name))
        out_log["Inputs"].append(str(input_out_path / out_name))
        out_log["raw_original"].append(str(fname))
        cv2.imwrite(str(label_out_path / out_name), label)
        cv2.imwrite(str(input_out_path / out_name), out_img)
        i += 1

# save log file as json
with open(str(out_path /"out_log.json"), "w") as js:
    json.dump(dict(out_log), js)



'''
for vid in vid_path.glob("*.mp4"):
    break
    if vid in processed_vids: # only process certain videos
        cap = cv2.VideoCapture(str(vid))

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
'''
