import os
import time
import datetime
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
dataset = Youtube_Greenscreen()

# net = ICNet(nclass = 2, backbone='resnet50', pretrained_base=False).to(device)
# net = FCN_Res50()
# net = modeling.deeplabv3_mobilenet(num_classes=2, pretrained_backbone=True)
# net = deeplabv3plus_mobilenet(num_classes=2, pretrained_backbone=True)
net = Deep_mobile_lstmV2()
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
    running_loss = 0
    batch_count = 0
    for batch in train_loader:
        images, labels = batch
        pred = net(images)
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
#
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
