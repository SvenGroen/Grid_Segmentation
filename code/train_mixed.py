import json
import argparse
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import nn
import torch
import numpy as np
import torchvision
import torch.optim as optim
import torchvision.transforms as T
from tqdm import tqdm
from models.custom.simple_models.simple_models import *
from models.custom.simple_models.UNet import *
from DataLoader.Datasets.Examples_Green.NY.NY_mixed import *
from models.DeepLabV3PlusPytorch.network import *
from models.ICNet.models import ICNet
from models.ICNet.utils import ICNetLoss, IterationPolyLR, SegmentationMetric, SetupLogger
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
import time
import sys
start_time = time.time()

config = {
    # DEFAULT CONFIG
    "model": "ICNet",
    # Options available: "UNet", "Deep_Res101", "ConvSame_3", "Deep_Res50", "Deep+_mobile", "ICNet"
    "ID": "01",
    "lr": 1e-02,
    "batch_size": 2,
    "num_epochs": 1,
    "scheduler_step_size": 15,
    "save freq": 1,
    "save_path": "code/models/trained_models/Examples_Green/multiples"
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

# selects model
if config["model"] == "UNet":
    net = UNet(in_channels=3, out_channels=2, n_class=2, kernel_size=3, padding=1, stride=1)
    net.train()
elif config["model"] == "Deep+_mobile":
    net = modeling.deeplabv3_mobilenet(num_classes=2,
                                       pretrained_backbone=True)  # https://github.com/VainF/DeepLabV3Plus-Pytorch
    net.train()
elif config["model"] == "Deep_Res101":
    net = Deeplab_Res101()
    norm_ImageNet = False
    net.train()
elif config["model"] == "Deep_Res50":
    net = Deeplab_Res50()
    norm_ImageNet = False
    net.train()
elif config["model"] == "ConvSame_3":
    net = ConvSame_3_net()
    net.train()
elif config["model"] == "ICNet":
    net = ICNet(nclass=2, backbone='resnet50', pretrained_base=False)  # https://github.com/liminn/ICNet-pytorch
    criterion = ICNetLoss()
    criterion.to(device)
    net.train()
else:
    net = None
    print("Model unknown")
net.to(device)

# parameters not set by config
if config["model"] != "ICNet":
    criterion = F.cross_entropy
norm_ImageNet = False
start_epoch = 0
optimizer = optim.Adam(net.parameters(), lr=config["lr"])
scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=config["scheduler_step_size"], gamma=0.1)
runtime = time.time()-start_time

# Dataset used (Greenscreen frames 512x270 (2048x1080 /4)
transform = [T.RandomPerspective(distortion_scale=0.1), T.ColorJitter(0.5, 0.5, 0.5),
             T.RandomAffine(degrees=10, scale=(1, 1.5)), T.RandomGrayscale(p=0.1), T.RandomGrayscale(p=0.1),
             T.RandomHorizontalFlip(p=0.7)]
dataset = NY_mixed(transforms=transform)  # <--- SET DATASET
train_loader = DataLoader(dataset=dataset, batch_size=config["batch_size"])

# saving the models
train_name = config["model"] + "_bs" + str(config["batch_size"]) + "_startLR" + format(config["lr"],
                                                                                       ".0e") + "Sched_Step_" + str(
    config["scheduler_step_size"]) + "ID" + config["ID"]  # sets name of model based on parameters
model_save_path = Path.cwd() / Path(config["save_path"]) / train_name
model_save_path.mkdir(parents=True, exist_ok=True)  # create folder to save results
with open(str(model_save_path / "train_config.json"), "w") as js:  # save learn config
    json.dump(config, js)

# Flags
LOAD_CHECKPOINT = True
LOAD_POSITION = -1
# Load previous model state if needed
lrs = []

if LOAD_CHECKPOINT:
    print("Trying to load previous Checkpoint ...")
    try:
        checkpoint = torch.load(str(model_save_path / train_name ) + ".pth.tar", map_location=torch.device(device))
        print("=> Loading checkpoint at epoch {}".format(checkpoint["epoch"][LOAD_POSITION]))
        net.load_state_dict(checkpoint["state_dict"][LOAD_POSITION])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"][LOAD_POSITION])
        start_epoch = checkpoint["epoch"][LOAD_POSITION]
        loss_values = checkpoint["loss_values"]
        lrs = checkpoint["lr"]
        scheduler.load_state_dict(checkpoint["scheduler"])
        for param_group in optimizer.param_groups:
            param_group['lr'] = lrs[LOAD_POSITION]
        config["batch_size"] = checkpoint["batchsize"][LOAD_POSITION]
        runtime = checkpoint["runtime"]
    except IOError:
        loss_values = []
        print("=> No previous checkpoint found")
        checkpoint = defaultdict(list)
        checkpoint["state_dict"].append(net.state_dict())
        checkpoint["optimizer_state_dict"].append(optimizer.state_dict())
        checkpoint["epoch"].append(start_epoch)
        checkpoint["lr"].append(config["lr"])
        checkpoint["batchsize"].append(config["batch_size"])
        checkpoint["loss_values"] = loss_values
        checkpoint["runtime"] = time.time() - start_time
        checkpoint["scheduler"] = scheduler.state_dict()
else:
    print("Previous checkpoints may exists but are not loaded")
    loss_values = []
    checkpoint = defaultdict(list)
    checkpoint["state_dict"].append(net.state_dict())
    checkpoint["optimizer_state_dict"].append(optimizer.state_dict())
    checkpoint["epoch"].append(start_epoch)
    checkpoint["lr"].append(config["lr"])
    checkpoint["batchsize"].append(config["batch_size"])
    checkpoint["loss_values"] = loss_values
    checkpoint["runtime"] = time.time() - start_time
    checkpoint["scheduler"] = scheduler.state_dict()

# Start training

print("--- Learning parameters: ---")
print("Device: {}; Model: {};\nLearning rate: {}; Number of epochs: {}; Batch Size: {}".format(device, config["model"],
                                                                                               str(config["lr"]),
                                                                                               str(config[
                                                                                                       "num_epochs"]),
                                                                                               str(config[
                                                                                                       "batch_size"])))
print("Loss criterion: ", criterion)
print("Applied Augmentation Transforms: ", transform)
print("Normalized by Imagenet_Values: ", norm_ImageNet)
print("Model {} saved at: {}".format(config["model"], str(model_save_path / train_name) ))
print("----------------------------")


# usefull functions
def save_figure(values, what=""):
    plt.plot(values)
    plt.xlabel("Epoch")
    plt.ylabel(what)
    plt.savefig(str(model_save_path/ train_name ) + "_" + what + ".jpg")
    plt.close()
    pass


def save_checkpoint(state, filename=str(model_save_path/ train_name ) + ".pth.tar"):
    print("=> Saving checkpoint at epoch {}".format(state["epoch"][-1]))
    checkpoint["state_dict"].append(net.state_dict())
    checkpoint["optimizer_state_dict"].append(optimizer.state_dict())
    checkpoint["epoch"].append(epoch)
    for param_group in optimizer.param_groups:
        checkpoint["lr"].append(param_group['lr'])
    checkpoint["batchsize"].append(config["batch_size"])
    checkpoint["loss_values"] = loss_values
    checkpoint["runtime"] = time.time() - start_time
    checkpoint["scheduler"] = scheduler.state_dict()
    torch.save(state, Path(filename))

time_tmp = []
start_train_time = time.time()
avrg_epoch_time = 60
restart_time =  60*60*4  
max_time = 60*60 * 24

print(">>>Start of Training<<<")


def restart_script():
    from subprocess import call
    VRAM = "3.4"
    if "Deep_Res" in config["model"]:
        VRAM = "3.9"
    recallParameter = 'qsub -N '+ "log_" + config["model"] +"_ep" + str(epoch) +' -l nv_mem_free='+VRAM+ ' -v CFG=' + str(model_save_path / "train_config.json") + ' train_mixed.sge'
    call(recallParameter, shell=True)
    pass



for epoch in tqdm(range(start_epoch, config["num_epochs"])):
    epoch_start = time.time()
    if epoch_start - start_time > max_time:
        sys.stderr.write("Stopping because programm was running to long ({} seconds > {})".format(epoch_start - start_time, max_time))
        break
    if epoch_start -start_time > restart_time - avrg_epoch_time:
        sys.stderr.write("Stopping at epoch {} because wall time would be reached".format(epoch))
        restart_script()
        break
    for param_group in optimizer.param_groups:
        lrs.append(param_group['lr'])
    running_loss = 0
    for batch in train_loader:
        images, labels = batch
        pred = net(images)
        loss = criterion(pred, labels.long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    loss_values.append(running_loss / len(dataset))
    scheduler.step()
    save_figure(loss_values, what="loss")
    save_figure(lrs, what="LR")
    if epoch % config["save_freq"] == 0:
        save_checkpoint(checkpoint)
        print("\nepoch: {},\t loss: {}".format(epoch, running_loss))

    epoch_end = time.time() - epoch_start
    time_tmp.append(epoch_end)
    avrg_epoch_time = np.array(time_tmp).mean()


print(">>>End of Training<<<")
# save model after training
save_checkpoint(checkpoint)
print("End of Python Script")
