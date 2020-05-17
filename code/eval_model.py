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
from models.ICNet.models import ICNet
from models.ICNet.utils import ICNetLoss, IterationPolyLR, SegmentationMetric, SetupLogger
from DataLoader.Datasets.Examples_Green.NY.NY_mixed import *
from DataLoader.Datasets.Examples_Green.NY.NY_mixed_HD import *
from PIL import Image
from pathlib import Path

print("---Start of Python File---")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

model = "Deep+_mobile"  # Options available: "UNet", "Deep_Res101", "ConvSame_3", "Deep_Res50", "Deep+_mobile" <--CHANGE
model_name = Path("Deep+_mobile_bs2_startLR1e-02Sched_Step_10ID0")  # <--CHANGE

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
elif model == "ICNet":
    net = ICNet(nclass=2, backbone='resnet50', pretrained_base=False)  # https://github.com/liminn/ICNet-pytorch
    criterion = ICNetLoss()
    criterion.to(device)
    net.train()
else:
    net = None
    print("Model unknown")

model_save_path = Path("code/models/trained_models/Examples_Green/multiples/session02") / model_name

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
dataset_HD = NY_mixed_HD()
train_loader_HD = DataLoader(dataset=dataset_HD, batch_size=batch_size, shuffle=True)


for loader in [train_loader,train_loader_HD]:

    to_PIL = transforms.ToPILImage()
    tmp_img, tmp_lbl, tmp_pred = [], [], []
    metrics = defaultdict(list)
    logger = defaultdict(list)

    if torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

    for batch in loader:
        images, labels = batch
        images.to(device)
        labels.to(device)
        if torch.cuda.is_available():
            start.record()
        start_time = time.time()
        with torch.autograd.profiler.profile(use_cuda=torch.cuda.is_available()) as prof:
            outputs = net(images)
            if model == "ICNet":
                outputs = outputs[0][:, :, :-2, :]
            outputs = torch.argmax(outputs, dim=1).float()
        logger["profiler_averages"].append(prof.total_average())
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
        else:
            break


    metrics["Mean-IoU"] = [np.array(metrics["IoU"]).mean()]

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

    with open(str(model_save_path / "eval_results.txt"), "w") as txt_file:
        txt_file.write("Model: {}\n".format(model_name))
        txt_file.write("Torch cudnn version: {}\n".format(torch.backends.cudnn.version()))
        txt_file.write("Torch cudnn enabled: {}\n".format(torch.backends.cudnn.enabled))
        txt_file.write("Cudnn benchmark: {}\n".format(torch.backends.cudnn.benchmark))
        txt_file.write("Cudnn deterministic: {}\n".format( torch.backends.cudnn.deterministic))
        if torch.cuda.is_available():
            txt_file.write("Mean cuda_time: {}\n".format( np.array(metrics["cuda_time"]).mean()))
        txt_file.write("Mean-IoU: {}; Mean_time.time() (in Secounds): {}\n".format(metrics["Mean-IoU"][0],
                       np.array(metrics["time.time()"]).mean()))
        txt_file.write("---Profiler Information---")
        txt_file.write("total average(): {}\n".format(logger["profiler_averages"][-1]))
        txt_file.write(prof.table(sort_by="self_cpu_time_total"))
        txt_file.write("\n--------------------------------------\n")

print("---Python file Completed---")



