from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import nn
import torch
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

print("Python Script Start")

model = "Deep_Res50"  # Options available: "UNet", "Deep_Res101", "ConvSame_3", "Deep_Res50", "Deep+_mobile", "ICNet"
ID = "01"
criterion = F.cross_entropy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
norm_ImageNet = False
if model == "UNet":
    net = UNet(in_channels=3, out_channels=2, n_class=2, kernel_size=3, padding=1, stride=1)
    net.train()
elif model == "Deep+_mobile":
    net = modeling.deeplabv3_mobilenet(num_classes=2,
                                       pretrained_backbone=True)  # https://github.com/VainF/DeepLabV3Plus-Pytorch
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
    net = ConvSame_3_net()
    net.train()
elif model == "ICNet":
    net = ICNet(nclass=2, backbone='resnet50', pretrained_base=False)  # https://github.com/liminn/ICNet-pytorch
    criterion = ICNetLoss()
    criterion.to(device)
    net.train()
else:
    net = None
    print("Model unknown")

net.to(device)

# --- Set Training parameters
# Augmentation transforms
transform = [T.RandomPerspective(distortion_scale=0.1), T.ColorJitter(0.5, 0.5, 0.5),
             T.RandomAffine(degrees=10, scale=(1, 1.5)), T.RandomGrayscale(p=0.1), T.RandomGrayscale(p=0.1),
             T.RandomHorizontalFlip(p=0.7)]
dataset = NY_mixed(transforms=transform)  # <--- SET DATASET
batch_size = 2  # <--- SET BATCHSIZE
if model == "Deep_Res101" or model == "Deep_Res50":
    assert batch_size > 1, "Batch size must be larger 1 for Deeplab to work"
lr = 1e-02  # <--- SET LEARNINGRATE
num_epochs = 100  # <--- SET NUMBER OF EPOCHS
scheduler_step_size = 15
start_epoch = 0
save_freq = 1

train_loader = DataLoader(dataset=dataset, batch_size=batch_size)
train_name = model + "_bs" + str(batch_size) + "_startLR" + format(lr, ".0e") +"Sched_Step_" +str(scheduler_step_size)+ "_cross_entropy" + "_ImageNet_" + str(norm_ImageNet) + "ID"+ID  # sets name of model based on parameters
model_save_path = Path(
    "code/models/trained_models/Examples_Green")  # <--- SET PATH WHERE MODEL WILL BE SAVED
model_save_path = Path.cwd() / model_save_path / train_name
model_save_path.mkdir(parents=True, exist_ok=True)  # create folder to save results
model_save_path = model_save_path / train_name

# Training Parameters
optimizer = optim.Adam(net.parameters(), lr=lr)


def save_checkpoint(state, filename=str(model_save_path) + ".pth.tar"):
    print("=> Saving checkpoint at epoch {}".format(state["epoch"][-1]))
    torch.save(state, Path(filename))


# Flags
LOAD_CHECKPOINT = True
LOAD_POSITION = -1
# Load previous model state if needed
lrs = []

if LOAD_CHECKPOINT:
    print("Trying to load previous Checkpoint ...")
    try:
        checkpoint = torch.load(str(model_save_path) + ".pth.tar", map_location=torch.device(device))
        print("=> Loading checkpoint at epoch {}".format(checkpoint["epoch"][LOAD_POSITION]))
        net.load_state_dict(checkpoint["state_dict"][LOAD_POSITION])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"][LOAD_POSITION])
        start_epoch = checkpoint["epoch"][LOAD_POSITION]
        loss_values = checkpoint["loss_values"]
        lrs = checkpoint["lr"]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lrs[LOAD_POSITION]
        batch_size = checkpoint["batchsize"][LOAD_POSITION]
    except IOError:
        loss_values = []
        print("=> No previous checkpoint found")
        checkpoint = defaultdict(list)
        checkpoint["state_dict"].append(net.state_dict())
        checkpoint["optimizer_state_dict"].append(optimizer.state_dict())
        checkpoint["epoch"].append(start_epoch)
        checkpoint["lr"].append(lr)
        checkpoint["batchsize"].append(batch_size)
        checkpoint["loss_values"] = loss_values
else:
    print("Previous checkpoints may exists but are not loaded")
    loss_values = []
    checkpoint = defaultdict(list)
    checkpoint["state_dict"].append(net.state_dict())
    checkpoint["optimizer_state_dict"].append(optimizer.state_dict())
    checkpoint["epoch"].append(start_epoch)
    checkpoint["lr"].append(lr)
    checkpoint["batchsize"].append(batch_size)
    checkpoint["loss_values"] = loss_values

# Start training

print("--- Learning parameters: ---")
print("Device: {}; Model: {};\nLearning rate: {}; Number of epochs: {}; Batch Size: {}".format(device, model, str(lr),
                                                                                               str(num_epochs),
                                                                                               str(batch_size)))
print("Loss criterion: ", criterion)
print("Applied Augmentation Transforms: ", transform)
print("Normalized by Imagenet_Values: ", norm_ImageNet)
print("Model {} saved at: {}".format(model, model_save_path))
print("----------------------------")


def save_figure(values, what=""):
    plt.plot(values)
    plt.xlabel("Epoch")
    plt.ylabel(what)
    plt.savefig(str(model_save_path) + "_" + what + ".jpg")
    plt.close()
    pass


# use schedular for learning rate
scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=scheduler_step_size, gamma=0.1)

print(">>>Start of Training<<<")
for epoch in tqdm(range(start_epoch, start_epoch + num_epochs)):
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
    if epoch % save_freq == 0:
        checkpoint["state_dict"].append(net.state_dict())
        checkpoint["optimizer_state_dict"].append(optimizer.state_dict())
        checkpoint["epoch"].append(epoch)
        for param_group in optimizer.param_groups:
            checkpoint["lr"].append(param_group['lr'])
        checkpoint["batchsize"].append(batch_size)
        checkpoint["loss_values"] = loss_values
        save_figure(loss_values, what="loss")
        save_figure(lrs, what="LR")
        save_checkpoint(checkpoint)
        print("\nepoch: {},\t loss: {}".format(epoch, running_loss))

print(">>>End of Training<<<")
# save model after training
save_checkpoint(checkpoint)
print("End of Python Script")
