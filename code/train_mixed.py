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
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict

print("Python Script Start")

model = "Deep+_mobile"  # Options available: "UNet", "Deep_Res101", "ConvSame_3", "Deep_Res50", "Deep+_mobile"

norm_ImageNet = False
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
else:
    net = None
    print("Model unknown")

# --- General Informations

print("--- Cuda information: ---")
if torch.cuda.is_available():
    print("cuda_current device: {}; device_count: {};\nis_available: {}; is_initialized: {};".format(
        torch.cuda.current_device(), torch.cuda.device_count(), torch.cuda.is_available(), torch.cuda.is_initialized()))
print("--- END OF CUDA INFORMATION ---")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)

# --- Set Training parameters
# Augmentation transforms
transform = [T.RandomPerspective(distortion_scale=0.1), T.ColorJitter(0.5, 0.5, 0.5),
             T.RandomAffine(degrees=10, scale=(1, 2)), T.RandomGrayscale(p=0.1), T.RandomGrayscale(p=0.1),
             T.RandomHorizontalFlip(p=0.7)]
dataset = NY_mixed(transforms=transform)  # <--- SET DATASET
batch_size = 2  # <--- SET BATCHSIZE
if model == "Deep_Res101" or model == "Deep_Res50":
    assert batch_size > 1, "Batch size must be larger 1 for Deeplab to work"
lr = 1e-03  # <--- SET LEARNINGRATE
num_epochs = 100  # <--- SET NUMBER OF EPOCHS
start_epoch = 0
save_freq = 20

train_loader = DataLoader(dataset=dataset, batch_size=batch_size)
train_name = model + "_bs" + str(batch_size) + "_lr" + format(lr, ".0e") + "_ep" + str(
    num_epochs) + "_cross_entropy" + "_ImageNet_" + str(norm_ImageNet)  # sets name of model based on parameters
model_save_path = Path(
    "code/models/trained_models/Examples_Green")  # <--- SET PATH WHERE MODEL WILL BE SAVED
model_save_path = Path.cwd() / model_save_path / train_name
model_save_path.mkdir(parents=True, exist_ok=True)  # create folder to save results
model_save_path = model_save_path / train_name
print("Model saved at: ", model_save_path)

# Training Parameters
optimizer = optim.Adam(net.parameters(), lr=lr)


def save_checkpoint(state, filename=str(model_save_path) + ".pth.tar"):
    print("=> Saving checkpoint at epoch {}".format(state["epoch"]))
    torch.save(state, Path(filename))


# Flags
LOAD_CHECKPOINT = True
LOAD_POSITION = -1
# Load previous model state if needed


if LOAD_CHECKPOINT:
    try:
        checkpoint = torch.load(str(model_save_path) + ".pth.tar", map_location=torch.device(device))
        print("=> Loading checkpoint at epoch {}".format(checkpoint["epoch"][LOAD_POSITION]))
        net.load_state_dict(checkpoint["state_dict"][LOAD_POSITION])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"][LOAD_POSITION])
        start_epoch = checkpoint["epoch"][LOAD_POSITION]
        loss_values = checkpoint["loss_values"]
        lr = checkpoint["lr"][LOAD_POSITION]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        batch_size = checkpoint["batchsize"][LOAD_POSITION]
    except IOError:
        loss_values = []
        print("No previous checkpoint found")
        checkpoint = defaultdict(list)
        checkpoint["state_dict"].append(net.state_dict())
        checkpoint["optimizer_state_dict"].append(optimizer.state_dict())
        checkpoint["epoch"].append(start_epoch)
        checkpoint["lr"].append(lr)
        checkpoint["batchsize"].append(batch_size)
        checkpoint["loss_values"] = loss_values
else:
    loss_values = []
    print("No previous checkpoint found")
    checkpoint = defaultdict(list)
    checkpoint["state_dict"].append(net.state_dict())
    checkpoint["optimizer_state_dict"].append(optimizer.state_dict())
    checkpoint["epoch"].append(start_epoch)
    checkpoint["lr"].append(lr)
    checkpoint["batchsize"].append(batch_size)
    checkpoint["loss_values"] = loss_values

# Start training
print("Start of Training")
print("--- Learning parameters: ---")
print("Device: {}; Model: {};\nLearning rate: {}; Number of epochs: {}; Batch Size: {}".format(device, model, str(lr),
                                                                                               str(num_epochs),
                                                                                               str(batch_size)))
print("Applied Augmentation Transforms: ", transform)
print("Normalized by Imagenet_Values: ", norm_ImageNet)
print("--- END OF LEARNING PARAMETERS ---")


def save_loss_figure(loss_values):
    plt.plot(loss_values)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(str(model_save_path) + "_loss.jpg")
    plt.close()
    pass


for epoch in tqdm(range(start_epoch, start_epoch + num_epochs)):
    running_loss = 0
    batch_count = 0
    for batch in train_loader:
        images, labels = batch
        pred = net(images)
        loss = F.cross_entropy(pred, labels.long())
        # loss = loss_criterion(pred, labels.long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_count += 1
        running_loss += loss.item() * images.size(0)
        running_loss += 2 * batch_count
    loss_values.append(running_loss / len(dataset))
    if epoch % save_freq == 0:
        checkpoint["state_dict"].append(net.state_dict())
        checkpoint["optimizer_state_dict"].append(optimizer.state_dict())
        checkpoint["epoch"].append(start_epoch)
        checkpoint["lr"].append(lr)
        checkpoint["batchsize"].append(batch_size)
        checkpoint["loss_values"] = loss_values
        save_loss_figure(loss_values)
        save_checkpoint(checkpoint)
        print("\nepoch: {}, \t batch: {}, \t loss: {}".format(epoch, batch_count, running_loss))

# save model after training
save_checkpoint(checkpoint)
