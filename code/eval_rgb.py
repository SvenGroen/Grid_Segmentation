import argparse
import json

import cv2
import torch
import torchvision.transforms as T
import numpy as np
import time
import matplotlib.pyplot as plt
import torch.nn.functional as F

from collections import defaultdict
from utils.stack import *
from utils.metrics import *
from models.custom.simple_models.simple_models import *
from models.custom.simple_models.UNet import *
from models.DeepLabV3PlusPytorch.network import *
from models.custom.simple_models.simple_models import ConvSame_3_net
from models.ICNet.models import ICNet
from torch.utils.data import DataLoader
from DataLoader.Datasets.Youtube.Youtube_Greenscreen import *
from DataLoader.Datasets.Youtube.Youtube_Greenscreen_mini import *
from DataLoader.Datasets.Youtube.background_dataset import *
from PIL import Image
from pathlib import Path

# load model name and model save path
# -mdl Deeplabv3Plus_rgb_lstmV1_bs8_startLR1e-02Sched_Step_6ID0 -pth code/models/trained_models/rgb_test
parser = argparse.ArgumentParser()
parser.add_argument("-mdl", "--model",
                    help="The name of the model.", type=str)
parser.add_argument("-pth", "--path", help="the path where the model is stored", type=str)

args = parser.parse_args()
model_name = args.model
model_path = Path(args.path)

print("---Start of Python File---")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)
output_size = (int(2048 / 4), int(1080 / 4))
FRAME_STOP_NUMBER = 29*15  # fps * dauer in sekunden in evaluation


def save_figure(values, path, what=""):
    plt.plot(values)
    plt.xlabel("Picture Number")
    plt.ylabel(what)
    plt.savefig(str(path / (what + ".jpg")))
    plt.close()
    pass


if "UNet" in model_name:
    net = UNet(in_channels=3, out_channels=2, n_class=2, kernel_size=3, padding=1, stride=1)
elif "Deep+_mobile" in model_name:
    net = Deeplabv3Plus_base(backbone="mobilenet")  # https://github.com/VainF/DeepLabV3Plus-Pytorch
elif "Deep_mobile_lstmV1" in model_name:
    net = Deeplabv3Plus_lstmV1(backbone="mobilenet")
elif "Deep_mobile_lstmV2" in model_name:
    net = Deeplabv3Plus_lstmV2(backbone="mobilenet")
elif "Deep_mobile_lstmV3" in model_name:
    net = Deeplabv3Plus_lstmV3(backbone="mobilenet")
elif "Deep_mobile_gruV1" in model_name:
    net = Deeplabv3Plus_gruV1(backbone="mobilenet")
elif "Deep_mobile_gruV2" in model_name:
    net = Deeplabv3Plus_gruV2(backbone="mobilenet")
elif "Deep_mobile_gruV3" in model_name:
    net = Deeplabv3Plus_gruV3(backbone="mobilenet")
elif "Deep+_resnet50" in model_name:
    net = Deeplabv3Plus_base(backbone="resnet50")
elif "Deep_resnet50_lstmV1" in model_name:
    net = Deeplabv3Plus_lstmV1(backbone="resnet50")
elif "Deep_resnet50_lstmV2" in model_name:
    net = Deeplabv3Plus_lstmV2(backbone="resnet50")
elif "Deep_resnet50_lstmV3" in model_name:
    net = Deeplabv3Plus_lstmV3(backbone="resnet50")
elif "Deep_resnet50_gruV1" in model_name:
    net = Deeplabv3Plus_gruV1(backbone="resnet50")
elif "Deep_resnet50_gruV2" in model_name:
    net = Deeplabv3Plus_gruV2(backbone="resnet50")
elif "Deep_resnet50_gruV3" in model_name:
    net = Deeplabv3Plus_gruV3(backbone="resnet50")
elif "Deeplabv3Plus_rgb_gru" in model_name:
    net = Deeplabv3Plus_rgb_gruV1()
elif "Deeplabv3Plus_rgb_lstmV1" in model_name:
    net = Deeplabv3Plus_rgb_lstmV1()
elif "Deeplabv3Plus_rgb" in model_name:
    net = Deeplabv3Plus_rgb()
elif "Deep_Res101" in model_name:
    net = Deeplab_Res101()
    norm_ImageNet = False
elif "Deep_Res50" in model_name:
    net = Deeplab_Res50()
    norm_ImageNet = False
elif "FCN_Res50" in model_name:
    net = FCN_Res50()
    norm_ImageNet = False
elif "ICNet" in model_name:
    net = ICNet(nclass=2, backbone='resnet50', pretrained_base=False)  # https://github.com/liminn/ICNet-pytorch
else:
    net = None
    print("Model unknown")

# load model and make directory to store evaluation results
model_name = Path(model_name)
full_path = model_path / model_name
eval_results_path = full_path / "evaluation_results"
eval_results_path.mkdir(parents=True, exist_ok=True)
print("Loading: " + str(full_path / model_name) + ".pth.tar")
LOAD_POSITION = -1
try:
    checkpoint = torch.load(str(full_path / model_name) + ".pth.tar", map_location=torch.device(device))
    print("=> Loading checkpoint at epoch {}".format(checkpoint["epoch"][LOAD_POSITION]))
    net.load_state_dict(checkpoint["state_dict"])
    print("Model was loaded.")
except IOError:
    print("model was not found")
    pass

# evaluation mode:
net.eval()
net.to(device)

# Load test data
dataset = Backgrounds(train=True)
# dataset = Youtube_Greenscreen_mini()
test_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=0)

# Meassure the metrics
metrics = defaultdict(AverageMeter)
metrics["Jaccard"] = AverageMeter()
metrics["Overall_acc"] = AverageMeter()
metrics["per_class_acc"] = AverageMeter()
metrics["dice"] = AverageMeter()
metrics["time_taken"] = AverageMeter()
metrics["cuda_time"] = AverageMeter()
if torch.cuda.is_available():
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

to_PIL = T.ToPILImage()  # for convertion into PILLOW Image
tmp_img, tmp_lbl, tmp_pred = [], [], []  # used for stacking images later

# Video writer to save results
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out_vid = cv2.VideoWriter(str(eval_results_path) + "/example_video.mp4", fourcc, 29, (1024, 270))

for i, batch in enumerate(test_loader):
    idx, (images, labels) = batch
    if idx.item() % 10 == 0:
        print("Processed {} from {} images".format(idx.item(), len(dataset)))

    # meassure time the model takes to predict
    if torch.cuda.is_available():
        start.record()
    start_time = time.time()
    pred = net(images)  # predict
    if torch.cuda.is_available():
        end.record()
        torch.cuda.synchronize(device=device)
        metrics["cuda_time"].append(start.elapsed_time(end))
    time_taken = time.time() - start_time

    # save metric results and old predictions
    metrics["time_taken"].update(time_taken)

    # Conversion for metric evaluations
    outputs = pred
    # overall_acc, avg_per_class_acc, avg_jacc, avg_dice = eval_metrics(outputs.to("cpu"),
    #                                                                   labels.to("cpu"),
    #                                                                   num_classes=3)
    overall_acc, avg_per_class_acc, avg_jacc, avg_dice = 0, 0, 0, 0
    metrics["IoU"].append(avg_jacc)
    metrics["Jaccard"].update(avg_jacc)
    metrics["Overall_acc"].update(overall_acc)
    metrics["per_class_acc"].update(avg_per_class_acc)
    metrics["dice"].update(avg_dice)

    # write videos
    # conversions since hstack expects PIL image or np array and cv2 np array with channel at last position
    tmp_prd = to_PIL(outputs.squeeze(0).cpu())
    tmp_inp = to_PIL(images.squeeze(0).cpu())
    tmp_inp = cv2.cvtColor(np.array(tmp_inp), cv2.COLOR_RGB2BGR)
    tmp_prd = cv2.cvtColor(np.array(tmp_prd), cv2.COLOR_RGB2BGR)
    stacked_out = hstack([tmp_inp, tmp_prd])
    out_vid.write(np.array(stacked_out))
    # stacked_out.save(str(eval_results_path / "example_output_{}.jpg".format(i)), "JPEG")
    # break after certain amount of frames (remove for final (last) evaluation)
    if i == FRAME_STOP_NUMBER:
        break

out_vid.release()

# save figures
save_figure(metrics["IoU"].history, path=eval_results_path, what="IoU")

# save metric results in .txt file
with open(eval_results_path / "metrics.json", "w") as js:
    tmp = {}
    for k, v in metrics.items():
        tmp[k] = v.avg.item() if torch.is_tensor(v.avg) else v.avg
    json.dump(dict(tmp), js)

with open(str(eval_results_path / "eval_results.txt"), "w") as txt_file:
    txt_file.write("Model: {}\n".format(model_name))
    txt_file.write("Cuda available: {}\n".format(torch.cuda.is_available()))
    txt_file.write("Torch cudnn version: {}\n".format(torch.backends.cudnn.version()))
    txt_file.write("Torch cudnn enabled: {}\n".format(torch.backends.cudnn.enabled))
    txt_file.write("Cudnn benchmark: {}\n".format(torch.backends.cudnn.benchmark))
    txt_file.write("Cudnn deterministic: {}\n".format(torch.backends.cudnn.deterministic))
    if torch.cuda.is_available():
        txt_file.write("Mean cuda_time: {}\n".format(metrics["cuda_time"].avg))
    txt_file.write("Mean Time taken: {}\n".format(metrics["time_taken"].avg))
    txt_file.write("Mean IoU: {}\n".format(metrics["Jaccard"].avg))
    txt_file.write("Mean Overall accuracy: {}\n".format(metrics["Overall_acc"].avg))
    txt_file.write("Mean per class accuracy: {}\n".format(metrics["per_class_acc"].avg))
    txt_file.write("Mean dice: {}\n".format(metrics["dice"].avg))
print("---Python file Completed---")
