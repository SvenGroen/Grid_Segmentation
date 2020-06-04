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

bgpath = Path("data/Images/Backgrounds/coco")
vid_path = Path(Path.cwd()) / "data/Videos/Examples_Green"
video_names = [vid.stem for vid in vid_path.glob("*")]
output_size = (int(2048 / 4), int(1080 / 4))
lower_green = np.array([0, 125, 0])
upper_green = np.array([100, 255, 120])
MAX_DURATION = 4

out_path = Path("data/Images/Greenscreen_Video_frames")
label_out_path = out_path / "labels"
input_out_path = out_path / "Input"
label_out_path.mkdir(parents=True, exist_ok=True)
input_out_path.mkdir(parents=True, exist_ok=True)
out_log = defaultdict(list)
i = 0
frame_counter = {}
for vid in video_names:
    frame_counter[vid] = 0
while len(video_names) != 0:
    print("--------------------------------")
    vid = np.random.choice(video_names)
    print("video: ", vid)
    print(*frame_counter.items(), sep=("\n"))
    cap = cv2.VideoCapture(str(vid_path / vid) +".mp4")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    current_frame_count = 0
    print("max frames: ", total_frames)
    if frame_counter[vid] >= total_frames:  # if the end of one video is reached, remove it from the list
        print("removing {} from {}, since all frames have been processed".format(vid, video_names))
        video_names.remove(vid)
        continue
    bgimg = cv2.imread(str(np.random.choice([img for img in bgpath.glob("*")])))
    bgimg = cv2.resize(bgimg, output_size)
    start = True
    while cap.isOpened():
        if frame_counter[vid] + MAX_DURATION * frame_rate == current_frame_count:  # if 4 seconds passed break
            frame_counter[vid] = current_frame_count
            print("4 seconds passed in ", vid)
            break
        ret, frame = cap.read()
        if ret:
            if current_frame_count != frame_counter[vid] and start:  # continue loading frames until the position of last stop
                current_frame_count += 1
                continue
            if start:
                tmp = current_frame_count
                print("starting at frame: ", current_frame_count)
            start = False
            frame = cv2.resize(frame, output_size)
            mask = cv2.inRange(frame, lower_green, upper_green)  # create a mask for the label and the background swap
            mask = np.expand_dims(mask, axis=-1)
            label = np.where(mask, 0, 255)
            out_img = np.where(mask, bgimg, frame)
            out_name = str(i).zfill(6) + ".jpg"
            out_log["labels"].append(str(label_out_path / out_name))
            out_log["Inputs"].append(str(input_out_path / out_name))
            cv2.imwrite(str(label_out_path / out_name), label)
            cv2.imwrite(str(input_out_path / out_name), out_img)
            current_frame_count += 1
            i += 1
        else:
            frame_counter[vid] = current_frame_count
            print("End of video: {} seconds passed in {}".format(int((current_frame_count - tmp) / frame_rate), vid))
            break
    cap.release()
# save log file as json
with open(str(out_path / "out_log.json"), "w") as js:
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
