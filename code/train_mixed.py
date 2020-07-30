import json
import argparse
import time
import sys

import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from PIL.Image import Image
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader
# import models and dataset
from utils import *
from models.custom.simple_models.simple_models import *
from models.custom.simple_models.UNet import *
from models.DeepLabV3PlusPytorch.network import *
from models.ICNet.models import ICNet
from models.ICNet.utils import ICNetLoss, IterationPolyLR, SegmentationMetric, SetupLogger
from DataLoader.Datasets.Youtube.YT_Greenscreen import *
from DataLoader.Datasets.Youtube.Youtube_Greenscreen import *
from DataLoader.Datasets.Youtube.Youtube_Greenscreen_mini import *
from utils.torch_poly_lr_decay import PolynomialLRDecay
import math

start_time = time.time()
sys.stderr.write("Starting at: {}\n".format(time.ctime(start_time)))

# -cfg code/models/trained_models/minisV2/Deep_mobile_lstm_bs4_startLR1e-03Sched_Step_20ID1/train_config.json
config = {  # DEFAULT CONFIG
    # This config is replaced by the config given as a parameter in -cfg, which will be generated in multiple_train.py.

    "model": "Deep_resnet50_lstmV5_1",
    "ID": "01",
    "lr": 1e-02,
    "batch_size": 4,
    "num_epochs": 6,
    "scheduler_step_size": 15,
    "save freq": 1,
    "loss": "CrossEntropy",  # / "SoftDice" / "Focal" / "CrossEntropy" / "Boundary" / "CrossDice"
    "save_path": "code/models/trained_models/testing",
    "track_ID": 99
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    sys.stderr.write(
        "Cuda is available: {},\nCuda device name: {},\nCuda device: {}\n".format(torch.cuda.is_available(),
                                                                                  torch.cuda.get_device_name(
                                                                                      0), device))
# Load Config
parser = argparse.ArgumentParser()
parser.add_argument("-cfg", "--config",
                    help="The Path to the configuration json for the model.\nShould include: model, ID, lr, batchsize,"
                         " num_epochs, scheduler_step_size, save_freq, save_path", type=str)
args = parser.parse_args()
if args.config is not None:
    with open(args.config) as js:
        print("Loading config: ", args.config)
        config = json.load(js)

# Assertions for config
if config["model"] == "Deep_Res101" or config["model"] == "Deep_Res50":
    assert config["batch_size"] > 1, "Batch size must be larger 1 for Deeplab to work"

# selects model based on config (find a better way to do this :D)
if config["model"] == "UNet":
    net = UNet(in_channels=3, out_channels=2, n_class=2, kernel_size=3, padding=1, stride=1)
elif config["model"] == "Deep+_mobile":
    net = Deeplabv3Plus_base(backbone="mobilenet")  # https://github.com/VainF/DeepLabV3Plus-Pytorch
    upper_lr_bound = 0.0002
    lower_lr_bound = upper_lr_bound / 6
elif config["model"] == "Deep_mobile_lstmV1":
    net = Deeplabv3Plus_lstmV1(backbone="mobilenet")
    upper_lr_bound = 0.002
    lower_lr_bound = upper_lr_bound / 6
elif config["model"] == "Deep_mobile_lstmV2_1":
    net = Deeplabv3Plus_lstmV2(backbone="mobilenet", activate_3d=False)
    upper_lr_bound = 0.002
    lower_lr_bound = upper_lr_bound / 6
elif config["model"] == "Deep_mobile_lstmV2_2":
    net = Deeplabv3Plus_lstmV2(backbone="mobilenet", activate_3d=True)
    upper_lr_bound = 0.002
    lower_lr_bound = upper_lr_bound / 6
elif config["model"] == "Deep_mobile_lstmV3":
    net = Deeplabv3Plus_lstmV3(backbone="mobilenet")
    upper_lr_bound = 0.0022
    lower_lr_bound = upper_lr_bound / 6
elif config["model"] == "Deep_mobile_lstmV4":
    net = Deeplabv3Plus_lstmV4(backbone="mobilenet")
    upper_lr_bound = 0.0021
    lower_lr_bound = upper_lr_bound / 6
elif config["model"] == "Deep_mobile_lstmV5_1":
    net = Deeplabv3Plus_lstmV5(backbone="mobilenet", keep_hidden=True)
    upper_lr_bound = 0.0023
    lower_lr_bound = upper_lr_bound / 6
elif config["model"] == "Deep_mobile_lstmV5_2":
    net = Deeplabv3Plus_lstmV5(backbone="mobilenet", keep_hidden=False)
    upper_lr_bound = 0.001
    lower_lr_bound = upper_lr_bound / 6
elif config["model"] == "Deep_mobile_gruV1":
    net = Deeplabv3Plus_gruV1(backbone="mobilenet")
    upper_lr_bound = 0.00055
    lower_lr_bound = upper_lr_bound / 6
elif config["model"] == "Deep_mobile_gruV2":
    net = Deeplabv3Plus_gruV2(backbone="mobilenet")
    upper_lr_bound = 0.001
    lower_lr_bound = upper_lr_bound / 6
elif config["model"] == "Deep_mobile_gruV3":
    net = Deeplabv3Plus_gruV3(backbone="mobilenet")
    upper_lr_bound = 0.00005
    lower_lr_bound = upper_lr_bound / 6
elif config["model"] == "Deep_mobile_gruV4":
    net = Deeplabv3Plus_gruV4(backbone="mobilenet")
    upper_lr_bound = 0.00025
    lower_lr_bound = upper_lr_bound / 6
elif config["model"] == "Deep+_resnet50":
    net = Deeplabv3Plus_base(backbone="resnet50")
    upper_lr_bound = 0.00055
    lower_lr_bound = upper_lr_bound / 6
elif config["model"] == "Deep_resnet50_lstmV1":
    net = Deeplabv3Plus_lstmV1(backbone="resnet50")
    upper_lr_bound = 0.002
    lower_lr_bound = upper_lr_bound / 6
elif config["model"] == "Deep_resnet50_lstmV2_1":
    net = Deeplabv3Plus_lstmV2(backbone="resnet50", activate_3d=False)
    upper_lr_bound = 0.002
    lower_lr_bound = upper_lr_bound / 6
elif config["model"] == "Deep_resnet50_lstmV2_2":
    net = Deeplabv3Plus_lstmV2(backbone="resnet50", activate_3d=True)
    upper_lr_bound = 0.002
    lower_lr_bound = upper_lr_bound / 6
elif config["model"] == "Deep_resnet50_lstmV3":
    net = Deeplabv3Plus_lstmV3(backbone="resnet50")
    upper_lr_bound = 0.00055
    lower_lr_bound = upper_lr_bound / 6
elif config["model"] == "Deep_resnet50_lstmV4":
    net = Deeplabv3Plus_lstmV4(backbone="resnet50")
    upper_lr_bound = 0.00025
    lower_lr_bound = upper_lr_bound / 6
elif config["model"] == "Deep_resnet50_lstmV5_1":
    net = Deeplabv3Plus_lstmV5(backbone="resnet50", keep_hidden=True)
    upper_lr_bound = 0.0023
    lower_lr_bound = upper_lr_bound / 6
elif config["model"] == "Deep_resnet50_lstmV5_2":
    net = Deeplabv3Plus_lstmV5(backbone="resnet50", keep_hidden=False)
    upper_lr_bound = 0.001
    lower_lr_bound = upper_lr_bound / 6
elif config["model"] == "Deep_resnet50_gruV1":
    net = Deeplabv3Plus_gruV1(backbone="resnet50")
    upper_lr_bound = 0.00055
    lower_lr_bound = upper_lr_bound / 6
elif config["model"] == "Deep_resnet50_gruV2":
    net = Deeplabv3Plus_gruV2(backbone="resnet50")
    upper_lr_bound = 0.001
    lower_lr_bound = upper_lr_bound / 6
elif config["model"] == "Deep_resnet50_gruV3":
    net = Deeplabv3Plus_gruV3(backbone="resnet50")
    upper_lr_bound = 0.000055
    lower_lr_bound = upper_lr_bound / 6
elif config["model"] == "Deep_resnet50_gruV4":
    net = Deeplabv3Plus_gruV4(backbone="resnet50")
    upper_lr_bound = 0.00026
    lower_lr_bound = upper_lr_bound / 6
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
    print("Model unknown")
net.train()
net.to(device)

# parameters not set by config
if config["model"] == "ICNet":
    criterion = ICNetLoss()
elif config["loss"] == "CrossEntropy":
    criterion = torch.nn.CrossEntropyLoss()
elif config["loss"] == "SoftDice":
    criterion = SegLoss.dice_loss.SoftDiceLoss(smooth=0.0001, apply_nonlin=F.softmax)
elif config["loss"] == "Focal":
    criterion = SegLoss.focal_loss.FocalLoss(smooth=0.0001, apply_nonlin=F.softmax)
elif config["loss"] == "Boundary":
    criterion = SegLoss.boundary_loss.BDLoss()
elif config["loss"] == "CrossDice":
    dice = SegLoss.dice_loss.SoftDiceLoss(smooth=0.0001, apply_nonlin=F.softmax)
    entropy = torch.nn.CrossEntropyLoss()
    criterion = lambda x, y: (dice(x, y) + entropy(x, y)) / 2.

norm_ImageNet = False
start_epoch = 0
# criterion.to(device)

batch_index = torch.tensor(range(config["batch_size"]))
# dataset = Youtube_Greenscreen(train=True, start_index=batch_index)
dataset = YT_Greenscreen(train=True, start_index=batch_index, batch_size=config["batch_size"])
# dataset = Youtube_Greenscreen_mini(start_index=batch_index, batch_size=config["batch_size"])
train_loader = DataLoader(dataset=dataset, batch_size=config["batch_size"], shuffle=False)

optimizer = optim.Adam(net.parameters(), lr=lower_lr_bound*1.15, weight_decay=0.0001)
# optimizer = optim.SGD(net.parameters(), lr=lower_lr_bound, weight_decay=0.0001, momentum=0.9)
# scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=config["scheduler_step_size"], gamma=0.1)
scheduler = PolynomialLRDecay(optimizer, max_decay_steps=config["num_epochs"], end_learning_rate=upper_lr_bound*0.01, power=2.0)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
sys.stderr.write(f"\nlen(loader) = {len(train_loader)}\n")
# scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=lower_lr_bound, max_lr=upper_lr_bound, cycle_momentum=False,
                                        # mode="triangular2", step_size_up=4 * len(train_loader))

# ----------------------------------------------------------------------------------------------------
# saving the models
train_name = config["model"] + "_bs" + str(config["batch_size"]) \
             + "_startLR" + format(config["lr"], ".0e") \
             + "Sched_Step_" + str(config["scheduler_step_size"]) \
             + "_" + config["loss"] + "_ID" + str(config["track_ID"])  # sets name of model based on parameters
model_save_path = Path.cwd() / Path(config["save_path"]) / train_name
model_save_path.mkdir(parents=True, exist_ok=True)  # create folder to save results
with open(str(model_save_path / "train_config.json"), "w") as js:  # save learn config
    json.dump(config, js)

# try to load previous model state
lrs = []
LOAD_POSITION = -1
print("Trying to load previous Checkpoint ...")
metric_log = defaultdict(list)
try:
    checkpoint = torch.load(str(model_save_path / train_name) + ".pth.tar", map_location=torch.device(device))
    print("=> Loading checkpoint at epoch {}".format(checkpoint["epoch"][LOAD_POSITION]))
    net.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"][LOAD_POSITION]
    loss_values = checkpoint["loss_values"]
    lrs = checkpoint["lr"]
    scheduler.load_state_dict(checkpoint["scheduler"])
    for param_group in optimizer.param_groups:
        param_group['lr'] = lrs[LOAD_POSITION]
    config["batch_size"] = checkpoint["batchsize"][LOAD_POSITION]
    runtime = checkpoint["runtime"]
    batch_index = checkpoint["batch_index"]
    metric_log = checkpoint["metric_log"]
except IOError:
    loss_values = []
    print("=> No previous checkpoint found")
    checkpoint = defaultdict(list)
    checkpoint["state_dict"] = net.state_dict()
    checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    checkpoint["epoch"].append(start_epoch)
    checkpoint["lr"] = lrs
    checkpoint["batchsize"].append(config["batch_size"])
    checkpoint["loss_values"] = loss_values
    checkpoint["runtime"] = time.time() - start_time
    checkpoint["scheduler"] = scheduler.state_dict()
    checkpoint["batch_index"] = batch_index
    checkpoint["metric_log"] = metric_log

# Start training

print("--- Learning parameters: ---")
print("Device: {}; Model: {};"
      "\nLearning rate: {}; Number of epochs: {}; Batch Size: {}".format(device, config["model"],

                                                                         str(config["lr"]),
                                                                         str(config[
                                                                                 "num_epochs"]),
                                                                         str(config[
                                                                                 "batch_size"])))
print("Loss criterion: ", criterion)
# print("Applied Augmentation Transforms: ", transform)
print("Normalized by Imagenet_Values: ", norm_ImageNet)
print("Model {} saved at: {}".format(config["model"], str(model_save_path / train_name)))
print("----------------------------")


# usefull functions
def save_figure(values, y_label="", x_label="Epoch"):
    # saves values in a plot
    plt.plot(values)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(str(model_save_path / train_name) + "_" + y_label + ".jpg")
    plt.close()
    pass


def save_checkpoint(checkpoint, filename=str(model_save_path / train_name) + ".pth.tar"):
    # saves model in a checkpoint for reloading
    sys.stderr.write("=> Saving checkpoint at epoch {}".format(checkpoint["epoch"][-1]))
    print("=> Saving checkpoint at epoch {}".format(checkpoint["epoch"][-1]))
    checkpoint["state_dict"] = net.state_dict()
    checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    checkpoint["epoch"].append(epoch)
    checkpoint["lr"] = lrs
    checkpoint["batchsize"].append(config["batch_size"])
    checkpoint["loss_values"] = loss_values
    checkpoint["runtime"] = time.time() - start_time
    checkpoint["scheduler"] = scheduler.state_dict()
    checkpoint["batch_index"] = batch_index
    checkpoint["running_loss"] = running_loss
    checkpoint["old_pred"] = old_pred
    checkpoint["metric_log"] = metric_log
    torch.save(checkpoint, Path(filename))


def restart_script():
    # restarts the training script
    from subprocess import call
    sys.stderr.write("restarting script ID: {}".format(str(config["track_ID"])))
    VRAM = "9G"
    recallParameter = 'qsub -N ' + "id" + str(config["track_ID"]) + "e" + str(epoch) + config[
        "model"] + ' -l nv_mem_free=' + VRAM + ' -v CFG=' + str(
        model_save_path / "train_config.json") + ' train_mixed.sge'
    call(recallParameter, shell=True)


def evaluate(model, train=False, eval_length=29 * 6, epoch=0, random_start=True):
    print("Evaluating")
    model.eval()
    model.start_eval()

    metrics = defaultdict(AverageMeter)
    to_PIL = T.ToPILImage()

    old_pred = [None, None]
    dset = YT_Greenscreen(train=train, start_index=batch_index, batch_size=1)
    if random_start:
        start_index = np.random.choice(range(len(dset) - eval_length))
        dset.set_start_index(int(start_index))
    loader = DataLoader(dataset=dset, batch_size=1, shuffle=False, num_workers=0)
    out_folder = model_save_path / "intermediate results"
    out_folder.mkdir(parents=True, exist_ok=True)
    mode = "train" if train else "val"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_vid = cv2.VideoWriter(str(out_folder) + "/intermediate_{}_ep{}.mp4".format(mode, epoch), fourcc, 29,
                              (1536, 270))
    for i, batch in enumerate(loader):
        sys.stderr.write("\nEvaluating\n")
        idx, video_start, (images, labels) = batch
        if torch.any(video_start.bool()):
            print(video_start)
            model.reset()
        pred = model(images, old_pred)  # predict
        outputs = torch.argmax(pred, dim=1).float()
        old_pred[0] = old_pred[1]  # oldest at 0 position
        old_pred[1] = pred.unsqueeze(1).detach()  # newest at 1 position
        # Conversion for metric evaluations
        labels = labels.type(torch.uint8)
        outputs = outputs.type(torch.uint8)
        overall_acc, avg_per_class_acc, avg_jacc, avg_dice = eval_metrics(outputs.to("cpu"),
                                                                          labels.to("cpu"),
                                                                          num_classes=2)

        metrics["Mean IoU"].update(avg_jacc)
        metrics["Pixel Accuracy"].update(overall_acc)
        metrics["Per Class Accuracy"].update(avg_per_class_acc)
        metrics["Dice"].update(avg_dice)

        # conversions since hstack expects PIL image or np array and cv2 np array with channel at last position

        tmp_prd = to_PIL(outputs[0].cpu().float())
        tmp_inp = to_PIL(images.squeeze(0).cpu())
        tmp_inp = Image.fromarray(cv2.cvtColor(np.asarray(tmp_inp), cv2.COLOR_RGB2BGR))
        tmp_lbl = to_PIL(labels.cpu().float())
        out_vid.write(np.array(stack.hstack([tmp_inp, tmp_lbl, tmp_prd])))
        # break after certain amount of frames (remove for final (last) evaluation)
        if i == eval_length:
            break
    out_vid.release()
    model.train()
    model.end_eval()
    return metrics


def visualize_metric(metric_log, step_size=2):
    for key in metric_log["train"][0]:
        y = defaultdict(list)
        for i in range(len(metric_log["train"])):
            y["train"].append(metric_log["train"][i][key].avg)
            y["test"].append(metric_log["test"][i][key].avg)
        print(y["train"], y["test"])
        plt.plot(range(0, epoch + 1, step_size), y["train"], color='red', label="train")
        plt.plot(range(0, epoch + 1, step_size), y["test"], color='blue', label="test")
        plt.legend()
        plt.title('Average Train/Test {} score'.format(key))
        plt.xlabel('Epoch')
        plt.ylabel('Average {}'.format(key))
        plt.savefig(str(model_save_path / "intermediate results" / (key + ".jpg")))
        plt.close()


print(">>>Start of Training<<<")
time_tmp = []
avrg_batch_time = 60 * 5
restart_time = 60 * 60 * 1.  # restart after 1 h
evaluation_steps = 10
restart = False  # flag
max_gpu_mem = 0
dataset.set_start_index(checkpoint["batch_index"])  # continue training at dataset position of last stop
epoch_start = time.time()
sys.stderr.write("\nEpoch starting at: {}".format(time.ctime(epoch_start)))

lrs_batch = []
for epoch in tqdm(range(start_epoch, config["num_epochs"])):

    running_loss = 0
    for batch in train_loader:
        batch_start_time = time.time()
        if batch_start_time + avrg_batch_time - start_time > restart_time:
            sys.stderr.write("\nStopping at epoch {} and batch_idx "
                             "{} because wall time would be reached".format(epoch, str(batch_index)))
            save_checkpoint(checkpoint)
            restart_script()
            restart = True
            break

        # no restart, continue training
        idx, video_start, (images, labels) = batch
        sys.stderr.write(f"\nCurrent epoch:{epoch}; \t current batch_idx: {idx}\n")
        if torch.any(video_start):
            net.reset()

        # check if end of batch is reached
        # the dataset will return 0-tensor as idx in case the end of the batch is reached
        if len(idx) == config["batch_size"]:
            if torch.all(idx == torch.zeros(config["batch_size"])):
                sys.stderr.write(f"\nEnd reached of batch at index {idx}\n")
                dataset.start_index = 0  # reset start index for the next batch
                break

        pred = net(images)

        if config["loss"] == "Boundary":
            mask = metrics.make_one_hot(labels.unsqueeze(1), C=2)
            loss = criterion(pred, labels, mask)
        else:
            loss = criterion(pred, labels)

        optimizer.zero_grad()
        # loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)

        # (detach so no error is thrown due to multiple backpropagations)
        batch_index = idx
        time_tmp.append(time.time() - batch_start_time)  # meassure time passed
        avrg_batch_time = np.array(time_tmp).mean()

        scheduler.step()
        lr_step = optimizer.state_dict()["param_groups"][0]["lr"]
        lrs_batch.append(lr_step)
        print(loss)
        break

    lrs.append(lr_step)
    if epoch % evaluation_steps == 0:
        print("evaluation at epoch", epoch)
        train_eval = evaluate(net, train=True, eval_length=29 * 2, epoch=epoch)
        test_eval = evaluate(net, train=False, eval_length=29 * 2, epoch=epoch)
        metric_log["train"].append(train_eval)
        metric_log["test"].append(test_eval)
        visualize_metric(metric_log, step_size=evaluation_steps)

    if restart:
        break

    # save figures and values at the end of the batch
    loss_values.append(running_loss / len(dataset))
    # scheduler.step(running_loss/len(dataset))
    save_figure(loss_values, y_label="loss")
    save_figure(lrs_batch, y_label="lr_each_steps", x_label="batch_steps")
    save_figure(lrs, y_label="LR")
    sys.stderr.write("End of Epoch: {}\n".format(epoch))

sys.stderr.write("End of Training\n")
print(">>>End of Training<<<")
# save model after training
if not restart:
    save_checkpoint(checkpoint)
print("End of Python Script")
