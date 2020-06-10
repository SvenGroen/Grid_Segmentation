import os
import time
import datetime
from pathlib import Path

import cv2
import yaml
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from DataLoader.Datasets.Examples_Green.NY.NY_mixed import *
from DataLoader.Datasets.Youtube.Youtube_Greenscreen import *
from models.custom.simple_models.simple_models import *
from models.DeepLabV3PlusPytorch.network import *
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# transform = [T.RandomPerspective(distortion_scale=0.1), T.ColorJitter(0.5, 0.5, 0.5),
#              T.RandomAffine(degrees=10, scale=(1, 2)), T.RandomGrayscale(p=0.1), T.RandomGrayscale(p=0.1),
#              T.RandomHorizontalFlip(p=0.7)]
# dataset = Youtube_Greenscreen()
dataset = Youtube_Greenscreen(train=True)
# # net = ICNet(nclass = 2, backbone='resnet50', pretrained_base=False).to(device)
net = FCN_Res50()
# # net = modeling.deeplabv3_mobilenet(num_classes=2, pretrained_backbone=True)
# # net = deeplabv3plus_mobilenet(num_classes=2, pretrained_backbone=True)
# net = Deep_mobile_lstmV2()
print(net)

# # criterion = ICNetLoss().to(device)
criterion = F.cross_entropy

batch_size = 4  # <--- SET BATCHSIZE
lr = 1e-03  # <--- SET LEARNINGRATE
num_epochs = 1  # <--- SET NUMBER OF EPOCHS

train_loader = DataLoader(dataset=dataset, batch_size=batch_size)
optimizer = optim.Adam(net.parameters(), lr=lr)
loss_values=[]
for epoch in tqdm(range(num_epochs)):
    old_pred = [None, None]
    running_loss = 0
    batch_count = 0
    for batch in train_loader:
        images, labels = batch
        pred = net(images, old_pred)
        loss = criterion(pred, labels.long())
        # loss = loss_criterion(pred, labels.long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_count += 1
        running_loss += loss.item() * images.size(0)
        running_loss += 2 * batch_count
        print(loss)
    loss_values.append(running_loss / len(dataset))

# import time
# # max_time = 30
# # avrg_time = 5
# # import sys
#
# # sys.stderr.write("a")
# # start_time = time.time()
# # while True:
# #     if time.time() -start_time > 30:
# #         print("break cause of wall")
# #     elif time.time() -start_time > max_time - avrg_time:
# #         print("break cause of avrg time")
# #     else:
# #         print(time.time() -start_time)
# #         time.sleep(1)
# #
#
# from collections import defaultdict
# a = defaultdict()
# a["test"] = 2
# print(a["test"])

from PIL import Image
import sys
import os
import numpy as np


# a = torch.rand(1, 100, 100)
# b = torch.rand(1, 100, 100)
# d = [a,b]
# # for i in range(len(d)):
# #     d[i] = d[i].unsqueeze(0)
# c = torch.cat([a]+d, dim=0)
#
# c=[a, None, None]
# c=[]
# print(None in c)


#
# def rgb_to_hsv(r, g, b):
#     maxc = max(r, g, b)
#     minc = min(r, g, b)
#     v = maxc
#     if minc == maxc:
#         return 0.0, 0.0, v
#     s = (maxc - minc) / maxc
#     rc = (maxc - r) / (maxc - minc)
#     gc = (maxc - g) / (maxc - minc)
#     bc = (maxc - b) / (maxc - minc)
#     if r == maxc:
#         h = bc - gc
#     elif g == maxc:
#         h = 2.0 + rc - bc
#     else:
#         h = 4.0 + gc - rc
#     h = (h / 6.0) % 1.0
#     return h, s, v
#
#
# GREEN_RANGE_MIN_HSV = (100, 80, 70)
# GREEN_RANGE_MAX_HSV = (185, 255, 255)
#
# output_size = (int(2048 / 4), int(1080 / 4))
#
# vid_path = Path(Path.cwd()) / "data/Videos/YT_originals/Group of girls goof off.mp4"
#
# cap = cv2.VideoCapture(str(vid_path))
#
# while cap.isOpened():
#     ret, frame = cap.read()
#
#     if ret:
#         frame = cv2.resize(frame, output_size)
#         hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#         h = hsv[:, :, 0]
#         h = hsv[:, :, 1]
#         h = hsv[:, :, 2]
#         cv2.imshow("a", frame)
