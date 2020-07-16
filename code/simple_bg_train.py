import json
import argparse
import time
import sys
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader
# import models and dataset
from models.custom.simple_models.simple_models import *
from models.custom.simple_models.UNet import *
from models.DeepLabV3PlusPytorch.network import *
from models.ICNet.models import ICNet
from models.ICNet.utils import ICNetLoss, IterationPolyLR, SegmentationMetric, SetupLogger
from DataLoader.Datasets.Youtube.Youtube_Greenscreen import *
from DataLoader.Datasets.Youtube.Youtube_Greenscreen_mini import *
from DataLoader.Datasets.Youtube.background_dataset import *
start_time = time.time()
sys.stderr.write("Starting at: {}\n".format(time.ctime(start_time)))

# -cfg code/models/trained_models/minisV2/Deep_mobile_lstm_bs4_startLR1e-03Sched_Step_20ID1/train_config.json
config = {  # DEFAULT CONFIG
    # This config is replaced by the config given as a parameter in -cfg, which will be generated in multiple_train.py.
    "model": "Deeplabv3Plus_rgb_gru",
    "ID": "01",
    "lr": 1e-02,
    "batch_size": 4,
    "num_epochs": 1,
    "scheduler_step_size": 15,
    "save freq": 1,
    "save_path": "code/models/trained_models/background"
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
elif config["model"] == "Deep_mobile_lstmV1":
    net = Deeplabv3Plus_lstmV1(backbone="mobilenet")
elif config["model"] == "Deep_mobile_lstmV2":
    net = Deeplabv3Plus_lstmV2(backbone="mobilenet")
elif config["model"] == "Deep_mobile_lstmV3":
    net = Deeplabv3Plus_lstmV3(backbone="mobilenet")
elif config["model"] == "Deep_mobile_gruV1":
    net = Deeplabv3Plus_gruV1(backbone="mobilenet")
elif config["model"] == "Deep_mobile_gruV2":
    net = Deeplabv3Plus_gruV2(backbone="mobilenet")
elif config["model"] == "Deep_mobile_gruV3":
    net = Deeplabv3Plus_gruV3(backbone="mobilenet")
elif config["model"] == "Deep+_resnet50":
    net = Deeplabv3Plus_base(backbone="resnet50")
elif config["model"] == "Deep_resnet50_lstmV1":
    net = Deeplabv3Plus_lstmV1(backbone="resnet50")
elif config["model"] == "Deep_resnet50_lstmV2":
    net = Deeplabv3Plus_lstmV2(backbone="resnet50")
elif config["model"] == "Deep_resnet50_lstmV3":
    net = Deeplabv3Plus_lstmV3(backbone="resnet50")
elif config["model"] == "Deep_resnet50_gruV1":
    net = Deeplabv3Plus_gruV1(backbone="resnet50")
elif config["model"] == "Deep_resnet50_gruV2":
    net = Deeplabv3Plus_gruV2(backbone="resnet50")
elif config["model"] == "Deep_resnet50_gruV3":
    net = Deeplabv3Plus_gruV3(backbone="resnet50")
elif config["model"] == "Deeplabv3Plus_rgb":
    net = Deeplabv3Plus_rgb()
elif config["model"] == "Deeplabv3Plus_rgb_gru":
    net = Deeplabv3Plus_rgb_gruV1()
elif config["model"] == "Deeplabv3Plus_rgb_lstmV1":
    net = Deeplabv3Plus_rgb_lstmV1(backbone="mobilenet")
else:
    net = None
    print("Model unknown")
net.train()
net.to(device)

# parameters not set by config
if config["model"] != "ICNet":
    criterion = F.mse_loss
norm_ImageNet = False
start_epoch = 0

optimizer = optim.Adam(net.parameters(), lr=config["lr"])
scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=config["scheduler_step_size"], gamma=0.1)

batch_index = torch.tensor(range(config["batch_size"]))
dataset = Backgrounds(train=True)
# dataset = Youtube_Greenscreen_mini()
train_loader = DataLoader(dataset=dataset, batch_size=config["batch_size"], shuffle=False)

# ----------------------------------------------------------------------------------------------------
# saving the models
train_name = config["model"] + "_bs" + str(config["batch_size"]) + "_startLR" + format(config["lr"],
                                                                                       ".0e") + "Sched_Step_" + str(
    config["scheduler_step_size"]) + "ID" + config["ID"]  # sets name of model based on parameters
model_save_path = Path.cwd() / Path(config["save_path"]) / train_name
model_save_path.mkdir(parents=True, exist_ok=True)  # create folder to save results
with open(str(model_save_path / "train_config.json"), "w") as js:  # save learn config
    json.dump(config, js)

# try to load previous model state
lrs = []
LOAD_POSITION = -1
print("Trying to load previous Checkpoint ...")
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
def save_figure(values, what=""):
    # saves values in a plot
    plt.plot(values)
    plt.xlabel("Epoch")
    plt.ylabel(what)
    plt.savefig(str(model_save_path / train_name) + "_" + what + ".jpg")
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
    torch.save(checkpoint, Path(filename))


def restart_script():
    # restarts the training script
    from subprocess import call
    VRAM = "4.5G"
    recallParameter = 'qsub -N ' + "ep" + str(epoch) + config["model"] + ' -l nv_mem_free=' + VRAM + ' -v CFG=' + str(
        model_save_path / "train_config.json") + ' train_rgb.sge'
    call(recallParameter, shell=True)
    pass


print(">>>Start of Training<<<")
time_tmp = []
avrg_batch_time = 60 * 5
restart_time = 60 * 60 * 0.48  # restart after 30 min
restart = False  # flag

dataset.set_start_index(checkpoint["batch_index"])  # continue training at dataset position of last stop
epoch_start = time.time()
sys.stderr.write("\nEpoch starting at: {}".format(time.ctime(epoch_start)))
for epoch in tqdm(range(start_epoch, config["num_epochs"])):
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
    lrs.append(lr)
    running_loss = 0

    for batch in train_loader:
        batch_start_time = time.time()
        if batch_start_time + avrg_batch_time - start_time > restart_time:
            sys.stderr.write("\nStopping at epoch {} and batch_idx "
                             "{} because wall time would be reached, with avg batch time of: {}".format(epoch, str(batch_index), str(avrg_batch_time)))
            save_checkpoint(checkpoint)
            restart_script()
            restart = True
            break

        # no restart, continue training
        idx, (images, labels) = batch

        # check if end of batch is reached
        # the dataset will return 0-tensor as idx in case the end of the batch is reached
        if torch.all(idx == torch.zeros(len(idx))):
            sys.stderr.write("\nEnd reached of batch")
            dataset.start_index = 0 # reset start index for the next batch
            break

        pred = net(images)
        loss = criterion(pred, labels)
        print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)

        # (detach so no error is thrown due to multiple backpropagations)
        batch_index = idx
        time_tmp.append(time.time() - batch_start_time) # meassure time passed
        avrg_batch_time = np.array(time_tmp).mean()

    if restart:
        break

    # save figures and values at the end of the batch
    loss_values.append(running_loss / len(dataset))
    scheduler.step()
    save_figure(loss_values, what="loss")
    save_figure(lrs, what="LR")
    sys.stderr.write("\nEnd of Epoch: {}\n".format(epoch))


sys.stderr.write("End of Training\n")
print(">>>End of Training<<<")
# save model after training
if not restart:
    save_checkpoint(checkpoint)
print("End of Python Script")
