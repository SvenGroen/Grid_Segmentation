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
from collections import defaultdict
from utils.metrics import get_IoU

net = ConvSame_3_net()
model_name = Path("ConvSame_3_net_bs5_lr1e-04_ep0_cross_entropy")
model_state_path = Path("code/models/custom/simple_models/trained_models/") / model_name

print("Loading: " + str(model_state_path / model_name) +".pth.tar")
try:
    if torch.cuda.is_available():
        checkpoint = torch.load(str(model_state_path / model_name) + ".pth.tar")
        net.load_state_dict(checkpoint["state_dict"])
    else:
        checkpoint = torch.load(str(model_state_path / model_name) + ".pth.tar", map_location=torch.device("cpu"))
        net.load_state_dict(checkpoint["state_dict"])
    print("Model was loaded.")
except IOError:
    print("model was not found")
    pass

# evaluation mode:
net.eval()
batch_size = 1
# Load test data
dataset = Example_NY()
train_loader = DataLoader(norm_ImageNet=False,dataset=dataset, batch_size=5, shuffle=True)

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
        img = to_PIL(images[0])
        lbl = to_PIL(labels[0])
        pred_img = to_PIL(outputs[0])
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
result.save(model_state_path / "output_example.jpg", "JPEG")
with open(model_state_path/ "metrics.json", "w") as js:
    json.dump(dict(metrics), js)
