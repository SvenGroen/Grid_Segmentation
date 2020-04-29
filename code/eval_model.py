import json
import torch
from models.custom.simple_models.simple_models import ConvSame_3_net
from DataLoader.Datasets.Examples.NY.NY import *
from pathlib import Path
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from utils.stack import *
from models.custom.simple_models.simple_models import *
from models.custom.simple_models.UNet import *
from DataLoader.Datasets.Examples.NY.NY import *
from collections import defaultdict
from utils.metrics import get_IoU

print("---Start of Python File---")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

model = "Deep_Res50"  # Options available: "UNet", "Deep_Res101", "ConvSame_3", "Deep_Res50" <--CHANGE
model_name = Path("Deep_Res50_bs2_lr5e-04_ep1000_cross_entropy_ImageNet_True")  # <--CHANGE

output_size = (1080, 2048)
# torchvision.models.segmentation.DeepLabV3(backbone=)
norm_ImageNet = False
if model == "UNet":
    net = UNet(in_channels=3, out_channels=2, n_class=2, kernel_size=3, padding=1, stride=1)
    net.train()
elif model == "Deep_Res101":
    net = Deeplab_Res101()
    norm_ImageNet = True
    output_size = (1080 / 2, 2048 / 2)
    net.train()
elif model == "Deep_Res50":
    net = Deeplab_Res50()
    norm_ImageNet = True
    output_size = (1080 / 4, 2048 / 4)
    net.train()
elif model == "ConvSame_3":
    net = ConvSame_3_net()  # <--- SET MODEL
else:
    print("Model unknown")

model_state_path = Path("code/models/custom/simple_models/trained_models/") / model_name

print("Loading: " + str(model_state_path / model_name) + ".pth.tar")
try:
    if torch.cuda.is_available():
        checkpoint = torch.load(str(model_state_path / model_name) + ".pth.tar")
        net.load_state_dict(checkpoint["state_dict"], strict=False)
    else:
        checkpoint = torch.load(str(model_state_path / model_name) + ".pth.tar", map_location=torch.device("cpu"))
        net.load_state_dict(checkpoint["state_dict"], strict=False)
    print("Model was loaded.")
except IOError:
    print("model was not found")
    pass

# evaluation mode:
net.eval()
net.to(device)
batch_size = 2
# Load test data
dataset = Example_NY(norm_ImageNet=norm_ImageNet, augmentation_transform=[transforms.CenterCrop(output_size)])
# dataset = Example_NY(norm_ImageNet=norm_ImageNet)
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

to_PIL = transforms.ToPILImage()
tmp_img, tmp_lbl, tmp_pred = [], [], []
metrics = defaultdict(list)
for i, batch in enumerate(train_loader):
    images, labels = batch
    outputs = net(images)
    outputs = torch.argmax(outputs, dim=1).float()
    metrics["IoU"].append(get_IoU(outputs, labels).tolist())

    # save example outputs
    if len(tmp_pred) < 5:
        img = to_PIL(images[0].to("cpu"))
        lbl = to_PIL(labels[0].to("cpu"))
        pred_img = to_PIL(outputs[0].to("cpu"))
        tmp_img.append(img)
        tmp_lbl.append(lbl)
        tmp_pred.append(pred_img)

metrics["Mean-IoU"] = [np.array(metrics["IoU"]).mean()]

# Save image file
out = []
for i in range(len(tmp_img)):
    out.append(hstack([tmp_img[i], tmp_lbl[i], tmp_pred[i]]))
result = vstack(out)
# out_folder = (model_state_path / model_name /"evaluation").mkdir(parents=True, exist_ok=True)

# save results
result.save(model_state_path / Path(model + "_example_output.jpg"), "JPEG")
with open(model_state_path / Path(model + "_metrics.json"), "w") as js:
    json.dump(dict(metrics), js)

print("---Python file Completed---")
