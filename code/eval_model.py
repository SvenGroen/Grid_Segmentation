import json
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
import time

from utils.stack import *
from models.custom.simple_models.simple_models import *
from models.custom.simple_models.UNet import *
from DataLoader.Datasets.Examples.NY.NY import *
from collections import defaultdict
from utils.metrics import get_IoU
from torch.utils.data import DataLoader
from models.DeepLabV3PlusPytorch.network import *
from models.custom.simple_models.simple_models import ConvSame_3_net
from DataLoader.Datasets.Examples_Green.NY.NY_mixed import *
from PIL import Image
from pathlib import Path

print("---Start of Python File---")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

model = "Deep_Res50"  # Options available: "UNet", "Deep_Res101", "ConvSame_3", "Deep_Res50", "Deep+_mobile" <--CHANGE
model_name = Path("Deep_Res50_bs2_lr1e-04_ep100_cross_entropy_ImageNet_True")  # <--CHANGE

# norm_ImageNet = False
if model == "UNet":
    net = UNet(in_channels=3, out_channels=2, n_class=2, kernel_size=3, padding=1, stride=1)
    net.train()
elif model == "Deep+_mobile":
    net = modeling.deeplabv3_mobilenet(num_classes=2, pretrained_backbone=True)
    net.train()
elif model == "Deep_Res101":
    net = Deeplab_Res101()
    norm_ImageNet = False
    net.train()
elif model == "Deep_Res50":
    net = Deeplab_Res50()
    norm_ImageNet = False
    net.train()
elif model == "ConvSame_3":
    net = ConvSame_3_net()  # <--- SET MODEL
    net.train()
else:
    net = None
    print("Model unknown")

model_save_path = Path("code/models/trained_models/Examples_Green/") / model_name

print("Loading: " + str(model_save_path / model_name) + ".pth.tar")
device = "cuda" if torch.cuda.is_available() else "cpu"
LOAD_POSITION = -1
batch_size = 2
try:
    checkpoint = torch.load(str(model_save_path / model_name) + ".pth.tar", map_location=torch.device(device))
    print("=> Loading checkpoint at epoch {}".format(checkpoint["epoch"][LOAD_POSITION]))
    net.load_state_dict(checkpoint["state_dict"][LOAD_POSITION])
    batch_size = checkpoint["batchsize"][LOAD_POSITION]
    print("Model was loaded.")
except IOError:
    print("model was not found")
    pass

# evaluation mode:
net.eval()
net.to(device)

# Load test data
transform = [T.RandomPerspective(distortion_scale=0.1), T.ColorJitter(0.5, 0.5, 0.5),
             T.RandomAffine(degrees=10, scale=(1, 2)), T.RandomGrayscale(p=0.1), T.RandomGrayscale(p=0.1),
             T.RandomHorizontalFlip(p=0.7)]
dataset = NY_mixed(transforms=transform)

train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

to_PIL = transforms.ToPILImage()
tmp_img, tmp_lbl, tmp_pred = [], [], []
metrics = defaultdict(list)

if torch.cuda.is_available():
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

for batch in train_loader:
    images, labels = batch
    images.to(device)
    labels.to(device)
    if torch.cuda.is_available():
        start.record()
    start_time = time.time()
    outputs = net(images)
    outputs = torch.argmax(outputs, dim=1).float()
    if torch.cuda.is_available():
        end.record()
        torch.cuda.synchronize(device=device)
        metrics["cuda_time"].append(start.elapsed_time(end))
    time_taken = time.time() - start_time
    metrics["time.time()"].append(time_taken)
    metrics["IoU"].append(get_IoU(outputs, labels).tolist())

    # save example outputs
    if len(tmp_pred) < 5:
        img = to_PIL(images[0].to("cpu"))
        lbl = to_PIL(labels[0].to("cpu"))
        pred_img = to_PIL(outputs[0].to("cpu"))
        tmp_img.append(img)
        tmp_lbl.append(lbl)
        tmp_pred.append(pred_img)
# print(prof.key_averages().table())

metrics["Mean-IoU"] = [np.array(metrics["IoU"]).mean()]
print("Mean-IoU: ", metrics["Mean-IoU"][0], "; Mean_time.time() (in Secounds): ", np.array(metrics["time.time()"]).mean())
if torch.cuda.is_available():
    print("Mean cuda_time: ", np.array(metrics["cuda_time"]).mean())
# Save image file
out = []
for i in range(len(tmp_img)):
    out.append(hstack([tmp_img[i], tmp_lbl[i], tmp_pred[i]]))
result = vstack(out)
# out_folder = (model_state_path / model_name /"evaluation").mkdir(parents=True, exist_ok=True)

# save results
result.save(model_save_path / Path(model + "_example_output.jpg"), "JPEG")
with open(model_save_path / Path(model + "_metrics.json"), "w") as js:
    json.dump(dict(metrics), js)

print("---Python file Completed---")
