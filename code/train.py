from torch.utils.data import DataLoader
from torch import nn
import torch
import torchvision
import torch.optim as optim
from tqdm import tqdm
from models.custom.simple_models.simple_models import ConvSame_3_net
from DataLoader.Datasets.Examples.NY.NY import *
from pathlib import Path
print("Python Script Start")


# Model, Dataset, train_loader, Learning Parameters
net = ConvSame_3_net()    # <--- SET MODEL
print("loading Dataset")
dataset =  Example_NY()                 # <--- SET DATASET
print("Dataset Loaded")
batch_size = 10                          # <--- SET BATCHSIZE
lr = 1e-4                               # <--- SET LEARNINGRATE
num_epochs = 1                         # <--- SET NUMBER OF EPOCHS


train_loader = DataLoader(dataset=dataset, batch_size=batch_size)
train_name = "ConvSame_3_net_bs" + str(batch_size) + "_lr" + str(lr) + "_ep" + str(num_epochs) + "_Version_2" # sets name of model based on parameters
model_save_path = Path("code/models/custom/simple_models/trained_models") # <--- SET PATH WHERE MODEL WILL BE SAVED
model_save_path = Path.cwd() / model_save_path / train_name

# Training Parameters
optimizer = optim.Adam(net.parameters(), lr=1e-4)
loss_criterion = nn.NLLLoss()

# Flags
LOAD_PREV_MODEL = False

print("Cuda information: ")
print("cuda_current device:{}; device_count:{}; is_available{}; is_initialized:{};".format(torch.cuda.current_device(),torch.cuda.device_count(), torch.cuda.is_available(), torch.cuda.is_initialized()))
print("END OF CUDA INFORMATION")


 

# Load previous model state if needed
if LOAD_PREV_MODEL:
    try:
        net.load_state_dict(torch.load(str(model_save_path)))
        print("Model {} was loaded.".format(train_name))
    except IOError:
        print("model was not found")
        pass

# Start training
print("Start of Training")
print("Learning with:")
print("Learning rate: {}, batch_size: {}, number of epochs: {}".format(lr,batch_size,num_epochs))
for epoch in tqdm(range(num_epochs)):
        total_loss = 0
        batch_count = 0
        for batch in train_loader:
            # print(batch_count)
            images, labels = batch
            # images = images.long()
            pred = net(images.float())
            # print(labels.shape)
            # labels = labels.squeeze(0).long()
            # print(labels.shape)
            # loss = F.cross_entropy(pred["out"], labels.long())
            loss = loss_criterion(pred, labels.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_count += 1
            total_loss += loss.item()
            
        print("\nepoch: {}, \t batch: {}, \t loss: {}".format(epoch, batch_count, total_loss))

# save model after training
torch.save(net.state_dict(), str(model_save_path))
