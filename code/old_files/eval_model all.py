import json

import cv2
import torch
import torchvision.transforms as T
import numpy as np
import time
import matplotlib.pyplot as plt

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
from DataLoader.Datasets.Youtube.Youtube_Greenscreen import *
from DataLoader.Datasets.Examples_Green.NY.NY_mixed import *
from DataLoader.Datasets.Examples_Green.NY.NY_mixed_HD import *
from PIL import Image
from pathlib import Path

print("---Start of Python File---")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)
output_size = (int(2048 / 4), int(1080 / 4))
model_path = Path("code/models/trained_models/LSTMs")


def save_figure(values, path, what=""):
    plt.plot(values)
    plt.xlabel("Epoch")
    plt.ylabel(what)
    plt.savefig(str(path / (what + ".jpg")))
    plt.close()
    pass


for model_name in model_path.glob("*"):
    print(model_name.stem)
    model_name = str(model_name)
    if "UNet" in model_name:
        net = UNet(in_channels=3, out_channels=2, n_class=2, kernel_size=3, padding=1, stride=1)
    elif "Deep_mobile_gruV2" in model_name:
        net = Deep_mobile_gruV2()
    elif "Deep_mobile_lstmV2" in model_name:
        net = Deep_mobile_lstmV2()
    elif "Deep_mobile_lstm" in model_name:
        net = Deep_mobile_lstm()
    elif "Deep_mobile_gru" in model_name:
        net = Deep_mobile_gru()
    elif "Deep+_mobile" in model_name:
        net = Deeplabv3Plus_mobile()  # https://github.com/VainF/DeepLabV3Plus-Pytorch
    elif "Deep_Res101" in model_name:
        net = Deeplab_Res101()
    elif "Deep_Res50" in model_name:
        net = Deeplab_Res50()
    elif "FCN_Res50" in model_name:
        net = FCN_Res50()
    elif "ConvSame_3" in model_name:
        net = ConvSame_3_net()  # <--- SET MODEL
    elif "ICNet" in model_name:
        net = ICNet(nclass=2, backbone='resnet50', pretrained_base=False)  # https://github.com/liminn/ICNet-pytorch
    else:
        net = None
        print("Model unknown")
    model_name = Path(model_name)
    full_path = model_path / model_name.stem
    eval_results_path = full_path / "evaluation_results"
    eval_results_path.mkdir(parents=True, exist_ok=True)
    print("Loading: " + str(full_path / model_name.stem) + ".pth.tar")
    LOAD_POSITION = -1
    try:
        checkpoint = torch.load(str(full_path / model_name.stem) + ".pth.tar", map_location=torch.device(device))
        print("=> Loading checkpoint at epoch {}".format(checkpoint["epoch"][LOAD_POSITION]))
        net.load_state_dict(checkpoint["state_dict"])
        print("Model was loaded.")
    except IOError:
        break
        print("model was not found")
        pass

    # evaluation mode:
    net.eval()
    net.to(device)

    # Load test data
    dataset = Youtube_Greenscreen(train=False)
    test_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=2)

    to_PIL = T.ToPILImage()
    tmp_img, tmp_lbl, tmp_pred = [], [], []
    metrics = defaultdict(list)
    logger = defaultdict(list)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_vid = cv2.VideoWriter(str(eval_results_path) + "/example_video.mp4", fourcc, 29, output_size)
    if torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

    for batch in test_loader:
        idx, (images, labels) = batch
        if idx.item() % 10 == 0:
            print("Processed {} from {} images".format(idx.item(), len(dataset)))
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
        # write video
        tmp = cv2.cvtColor(outputs.squeeze(0).numpy(), cv2.COLOR_GRAY2BGR)
        out_vid.write(np.uint8(tmp * 255.))
        # save example outputs
        if len(tmp_pred) < 5:
            out_vid.write(np.uint8(tmp))
            img = to_PIL(images[0].to("cpu"))
            lbl = to_PIL(labels[0].to("cpu"))
            pred_img = to_PIL(outputs[0].to("cpu"))
            tmp_img.append(img)
            tmp_lbl.append(lbl)
            tmp_pred.append(pred_img)

    out_vid.release()
    metrics["Mean-IoU"] = [np.array(metrics["IoU"]).mean()]

    # Save image file
    out = []
    for i in range(len(tmp_img)):
        out.append(hstack([tmp_img[i], tmp_lbl[i], tmp_pred[i]]))
    result = vstack(out)

    save_figure(metrics["IoU"], path=eval_results_path, what="IoU")

    # save results
    result.save(str(eval_results_path / "example_output.jpg"), "JPEG")
    with open(eval_results_path / "metrics.json", "w") as js:
        json.dump(dict(metrics), js)

    with open(str(eval_results_path / "eval_results.txt"), "w") as txt_file:
        txt_file.write("Model: {}\n".format(model_name))
        txt_file.write("Torch cudnn version: {}\n".format(torch.backends.cudnn.version()))
        txt_file.write("Torch cudnn enabled: {}\n".format(torch.backends.cudnn.enabled))
        txt_file.write("Cudnn benchmark: {}\n".format(torch.backends.cudnn.benchmark))
        txt_file.write("Cudnn deterministic: {}\n".format(torch.backends.cudnn.deterministic))
        if torch.cuda.is_available():
            txt_file.write("Mean cuda_time: {}\n".format(np.array(metrics["cuda_time"]).mean()))
        txt_file.write("Mean-IoU: {}; Mean_time.time() (in Secounds): {}\n".format(metrics["Mean-IoU"][0],
                                                                                   np.array(
                                                                                       metrics["time.time()"]).mean()))

print("---Python file Completed---")

