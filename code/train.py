from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import nn
import torch
import torchvision
import torch.optim as optim
from tqdm import tqdm
from models.custom.simple_models.simple_models import *
from models.custom.simple_models.UNet import *
from DataLoader.Datasets.Examples.NY.NY import *
from pathlib import Path

model = "Deep_Res50"  # Options available: "UNet", "Deep_Res101", "ConvSame_3", "Deep_Res50"
output_size = (1080,2048)
# torchvision.models.segmentation.DeepLabV3(backbone=)
norm_ImageNet = False 
if model == "UNet":
    net = UNet(in_channels=3, out_channels=2, n_class=2, kernel_size=3, padding=1, stride=1)
    net.train()
elif model == "Deep_Res101":
    net = Deeplab_Res101()
    norm_ImageNet = True
    output_size = (1080/2, 2048/2)
    net.train()
elif model == "Deep_Res50":
    net = Deeplab_Res50()
    norm_ImageNet = True
    output_size = (1080 / 4, 2048 / 4)
    net.train()

elif model == "ConvSame_3":
    net = ConvSame_3_net()  # <--- SET MODEL
else:
    print("Model unknown")

# --- General Informations
print("Python Script Start")
print("Cuda information: ")
if torch.cuda.is_available():
    print("cuda_current device: {}; device_count: {}; is_available: {}; is_initialized: {};".format(
        torch.cuda.current_device(), torch.cuda.device_count(), torch.cuda.is_available(), torch.cuda.is_initialized()))
print("END OF CUDA INFORMATION")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)
net.to(device)
# Model, Dataset, train_loader, Learning Parameters

dataset = Example_NY(norm_ImageNet=norm_ImageNet,
                     augmentation_transform=[transforms.CenterCrop(output_size)])  # <--- SET DATASET
batch_size = 2  # <--- SET BATCHSIZE
if model == "Deep_Res101":
    assert batch_size > 1, "Batch size must be larger 1 for Deeplab to work"
lr = 5e-03  # <--- SET LEARNINGRATE
num_epochs = 1001  # <--- SET NUMBER OF EPOCHS
start_epoch = 0
save_freq = 50

train_loader = DataLoader(dataset=dataset, batch_size=batch_size)
train_name = model + "_bs" + str(batch_size) + "_lr" + format(lr, ".0e") + "_ep" + str(
    num_epochs) + "_cross_entropy" + "_ImageNet_"+str(norm_ImageNet)  # sets name of model based on parameters
model_save_path = Path("code/models/custom/simple_models/trained_models")  # <--- SET PATH WHERE MODEL WILL BE SAVED
model_save_path = Path.cwd() / model_save_path / train_name
model_save_path.mkdir(parents=True, exist_ok=True)  # create folder to save results
model_save_path = model_save_path / train_name
print("Model saved at: ", model_save_path)

# Training Parameters
optimizer = optim.Adam(net.parameters(), lr=lr)
loss_criterion = nn.NLLLoss()


def save_checkpoint(state, filename=str(model_save_path) + ".pth.tar"):
    print("=> Saving checkpoint at epoch {}".format(state["epoch"]))
    torch.save(state, Path(filename))


# Flags
LOAD_CHECKPOINT = True
# Load previous model state if needed

if LOAD_CHECKPOINT:
    try:
        checkpoint = torch.load(str(model_save_path) + ".pth.tar")
        print("=> Loading checkpoint at epoch {}".format(checkpoint["epoch"]))
        net.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        total_loss = checkpoint["total_loss"]
        lr = checkpoint["lr"]
        batch_size = checkpoint["batchsize"]
    except IOError:
        print("No previous checkpoint found")
        checkpoint = {}
        checkpoint["state_dict"] = net.state_dict()
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        checkpoint["epoch"] = start_epoch
        checkpoint["lr"] = lr
        checkpoint["batchsize"] = batch_size
        checkpoint["total_loss"] = 0
else:
    print("No previous checkpoint found")
    checkpoint = {}
    checkpoint["state_dict"] = net.state_dict()
    checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    checkpoint["epoch"] = start_epoch
    checkpoint["lr"] = lr
    checkpoint["batchsize"] = batch_size
    checkpoint["total_loss"] = 0

# Start training
print("Start of Training")
print("Learning with:")
print("Learning rate: {}, batch_size: {}, number of epochs: {}".format(lr, batch_size, num_epochs))
for epoch in tqdm(range(start_epoch, start_epoch + num_epochs)):
    total_loss = 0
    batch_count = 0
    for batch in train_loader:
        # print(batch_count)
        images, labels = batch
        # if torch.cuda.is_available():
        #     pred=net(images.cuda())
        # else:
        #     pred=net(images.Float())
        pred = net(images)
        loss = F.cross_entropy(pred, labels.long())
        # loss = loss_criterion(pred, labels.long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_count += 1
        total_loss += loss.item()

    if epoch % save_freq == 0:
        checkpoint["state_dict"] = net.state_dict()
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        checkpoint["epoch"] = start_epoch
        checkpoint["lr"] = lr
        checkpoint["batchsize"] = batch_size
        checkpoint["total_loss"] = total_loss
        save_checkpoint(checkpoint)
        print("\nepoch: {}, \t batch: {}, \t loss: {}".format(epoch, batch_count, total_loss))

# save model after training
save_checkpoint(checkpoint)
