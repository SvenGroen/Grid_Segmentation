import glob
import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
import os
from collections import defaultdict
from pathlib import Path

print("\n")
print("Home Path: ", Path.home())
print("\n")
print("Home Path resolve(): ", Path.home().resolve())
print("\n")
print("Home Path absolute(): ", Path.home().absolute())
print("\n")
print("Current WD: ", Path.cwd())

p = Path("data/Videos/Examples")
p = p/"Asiatin_final.mp4"
print("Video-file: ", p)
cap = cv2.VideoCapture(str(p))

'''
for root, dirs, files in os.walk("./videos/Outputs"):
    video_frames = []
    video_masks = []
    video_labels = []
    for file in files:

        path = root + "/" + file
        cap = cv2.VideoCapture(path)
        frames = []
        masks = []
        labels = []
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame_width = int(cap.get(3))
                frame_height = int(cap.get(4))
                frames.append(frame)
                masks.append(np.where((frame == 0), False, True))
                label = (np.any(frame, axis=2, keepdims=False) * 1.).astype(np.uint8)
                labels.append(label)
            else:
                break
        cap.release()
        np.savez("./data/" + file[:-4]+ ".npz", frames=np.array(frames), masks=np.array(masks),
                 labels=np.array(labels))
        print("Finished Video: {}".format(file))
print("Saved files")

def get_all_frames(cap, dual_channel=False):
    frames = []
    labels = []
    while cap.isOpened():
        ret, frame = cap.read()
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        bg = cv2.imread("./images/Coco_val_2017/000000000139.jpg")
        bg = cv2.resize(bg, (frame_width, frame_height))
        image, label = change_background(bg, frame)

        # print(label.shape)
        # img.show()
        print(image.shape)
        print(label.shape)
        if not ret:
            break
        cv2.imshow("v", image)
        cv2.imshow("frame", label)
        # label = frame
        # label = Image.fromarray(label)
        # label = transform_gry(label)
        # if dual_channel:
        #     label = torch.cat((1 - label, label), dim=0)  # One-Hot-Encodes labels
        # frame = Image.fromarray(frame)
        # frame = transform(frame)
        # frames.append(frame)
        # labels.append(label)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    return frames, labels  # np.asarray(frames, dtype=np.uint8)

# a, b = get_all_frames(cap, False)
'''