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
from PIL import Image
from pathlib import Path

# load model name and model save path
parser = argparse.ArgumentParser()
parser.add_argument("-mdl", "--model",
                    help="The name of the model.", type=str)
parser.add_argument("-pth", "--path", help="the path where the model is stored", type=str)
#-mdl Deep_mobile_gruV3_bs6_startLR1e-01Sched_Step_6_SoftDice_ID0 -pth code/models/trained_models/minis


args = parser.parse_args()
model_name = args.model
model_path = Path(args.path)

with open(str(model_path/model_name/"train_config.json")) as js:
    config = json.load(js)

print("---Start of Python File---")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)
output_size = (int(2048 / 4), int(1080 / 4) * 2)
FRAME_STOP_NUMBER = 29 * 12  # fps * dauer in sekunden in evaluation
sys.stderr.write("\nModel: {}.\n".format(config["model"]))
if config["model"] == "Deep_mobile_lstmV5.1":
    config["model"]

def save_figure(values, path, what=""):
    plt.plot(values)
    plt.xlabel("Picture Number")
    plt.ylabel(what)
    plt.savefig(str(path / (what + ".jpg")))
    plt.close()
    pass


if config["model"] == "UNet":
    net = UNet(in_channels=3, out_channels=2, n_class=2, kernel_size=3, padding=1, stride=1)
elif config["model"] == "Deep+_mobile":
    net = Deeplabv3Plus_base(backbone="mobilenet")  # https://github.com/VainF/DeepLabV3Plus-Pytorch
elif config["model"] == "Deep_mobile_lstmV1":
    net = Deeplabv3Plus_lstmV1(backbone="mobilenet")
elif config["model"] == "Deep_mobile_lstmV2_1":
    net = Deeplabv3Plus_lstmV2(backbone="mobilenet", activate_3d=False, hidden_return_layer=0)
elif config["model"] == "Deep_mobile_lstmV2_2":
    net = Deeplabv3Plus_lstmV2(backbone="mobilenet", activate_3d=False, hidden_return_layer=-1)
elif config["model"] == "Deep_mobile_lstmV2_3":
    net = Deeplabv3Plus_lstmV2(backbone="mobilenet", activate_3d=True, hidden_return_layer=0)
elif config["model"] == "Deep_mobile_lstmV3":
    net = Deeplabv3Plus_lstmV3(backbone="mobilenet")
elif config["model"] == "Deep_mobile_lstmV4":
    net = Deeplabv3Plus_lstmV4(backbone="mobilenet")
elif config["model"] == "Deep_mobile_lstmV5_1":
    net = Deeplabv3Plus_lstmV5(backbone="mobilenet", keep_hidden=True)
elif config["model"] == "Deep_mobile_lstmV5_2":
    net = Deeplabv3Plus_lstmV5(backbone="mobilenet", keep_hidden=False)
elif config["model"] == "Deep_mobile_gruV1":
    net = Deeplabv3Plus_gruV1(backbone="mobilenet")
elif config["model"] == "Deep_mobile_gruV2":
    net = Deeplabv3Plus_gruV2(backbone="mobilenet")
elif config["model"] == "Deep_mobile_gruV3":
    net = Deeplabv3Plus_gruV3(backbone="mobilenet")
elif config["model"] == "Deep_mobile_gruV4":
    net = Deeplabv3Plus_gruV4(backbone="mobilenet")
elif config["model"] == "Deep+_resnet50":
    net = Deeplabv3Plus_base(backbone="resnet50")
elif config["model"] == "Deep_resnet50_lstmV1":
    net = Deeplabv3Plus_lstmV1(backbone="resnet50")
elif config["model"] == "Deep_resnet50_lstmV2":
    net = Deeplabv3Plus_lstmV2(backbone="resnet50")
elif config["model"] == "Deep_resnet50_lstmV3":
    net = Deeplabv3Plus_lstmV3(backbone="resnet50")
elif config["model"] == "Deep_resnet50_lstmV4":
    net = Deeplabv3Plus_lstmV4(backbone="resnet50")
elif config["model"] == "Deep_resnet50_lstmV5":
    net = Deeplabv3Plus_lstmV5(backbone="resnet50")
elif config["model"] == "Deep_resnet50_gruV1":
    net = Deeplabv3Plus_gruV1(backbone="resnet50")
elif config["model"] == "Deep_resnet50_gruV2":
    net = Deeplabv3Plus_gruV2(backbone="resnet50")
elif config["model"] == "Deep_resnet50_gruV3":
    net = Deeplabv3Plus_gruV3(backbone="resnet50")
elif config["model"] == "Deep_resnet50_gruV4":
    net = Deeplabv3Plus_gruV4(backbone="resnet50")
elif config["model"] == "Deep_Res101":
    net = Deeplab_Res101()
    norm_ImageNet = False
elif config["model"] == "Deep_Res50":
    net = Deeplab_Res50()
    norm_ImageNet = False
elif config["model"] == "FCN_Res50":
    net = FCN_Res50()
    norm_ImageNet = False
elif config["model"] == "ICNet":
    net = ICNet(nclass=2, backbone='resnet50', pretrained_base=False)  # https://github.com/liminn/ICNet-pytorch

else:
    net = None
    sys.stderr.write("\nModel {} unkown.\n".format(config["model"]))

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
    net.load_state_dict(checkpoint["state_dict"], strict=True)
    print("Model was loaded.")
except IOError:
    print("model was not found")
    pass

# evaluation mode:
net.eval()
net.to(device)

# Load test data
# dataset = Youtube_Greenscreen(train=False)
dataset = Youtube_Greenscreen_mini(batch_size=1)
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
fourcc2 = cv2.VideoWriter_fourcc(*"mp4v")
out_vid = cv2.VideoWriter(str(eval_results_path) + "/example_video.mp4", fourcc, 29, (1536, 270))
out_vid2 = cv2.VideoWriter(str(eval_results_path) + "/example_video2.mp4", fourcc2, 29, (512, 270))

old_pred = [None, None]
for i, batch in enumerate(test_loader):
    idx, (images, labels) = batch
    if idx.item() % 10 == 0:
        print("Processed {} from {} images".format(idx.item(), len(dataset)))

    # meassure time the model takes to predict
    if torch.cuda.is_available():
        start.record()
    start_time = time.time()
    pred = net(images, old_pred)  # predict
    # outputs = pred.float()
    outputs = torch.argmax(pred, dim=1).float()
    if torch.cuda.is_available():
        end.record()
        torch.cuda.synchronize(device=device)
        metrics["cuda_time"].append(start.elapsed_time(end))
    time_taken = time.time() - start_time

    # save metric results and old predictions
    metrics["time_taken"].update(time_taken)
    old_pred[0] = old_pred[1]  # oldest at 0 position
    old_pred[1] = pred.unsqueeze(1).detach()  # newest at 1 position

    # Conversion for metric evaluations
    labels = labels.type(torch.uint8)
    outputs = outputs.type(torch.uint8)
    overall_acc, avg_per_class_acc, avg_jacc, avg_dice = eval_metrics(outputs.to("cpu"),
                                                                      labels.to("cpu"),
                                                                      num_classes=2)
    metrics["IoU"].append(avg_jacc)
    metrics["Jaccard"].update(avg_jacc)
    metrics["Overall_acc"].update(overall_acc)
    metrics["per_class_acc"].update(avg_per_class_acc)
    metrics["dice"].update(avg_dice)

    # create mask for evaluation video 2 (raw image with Greenscreen based on prediction)
    mask = outputs.squeeze(0).cpu().numpy()
    mask = np.expand_dims(mask, axis=-1)

    # write videos
    # conversions since hstack expects PIL image or np array and cv2 np array with channel at last position
    tmp_prd = to_PIL(outputs[0].cpu().float())
    tmp_inp = to_PIL(images.squeeze(0).cpu())
    tmp_inp = Image.fromarray(cv2.cvtColor(np.asarray(tmp_inp), cv2.COLOR_RGB2BGR))
    tmp_lbl = to_PIL(labels.cpu().float())
    out_vid.write(np.array(hstack([tmp_inp, tmp_lbl, tmp_prd])))
    replaced_out = np.where(mask, tmp_inp, [0, 255, 0])
    out_vid2.write(np.uint8(replaced_out))
    stacked_out = hstack([tmp_inp, tmp_lbl, tmp_prd])
    if i == 27:
        # snapchot of one prediciton
        stacked_out.save(str(eval_results_path / "example_output.jpg"), "JPEG")
    # break after certain amount of frames (remove for final (last) evaluation)
    if i == FRAME_STOP_NUMBER:
        break
out_vid2.release()
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
