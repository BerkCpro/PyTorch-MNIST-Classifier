# %%
import torch
from torch import nn

import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from tqdm import tqdm

from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

from pathlib import Path

# %%

train_data = datasets.MNIST(root= "data",
                            train= True,
                            transform=ToTensor(),
                            download=True,
                            target_transform=None)
test_data = datasets.MNIST(root= "data",
                           train= False,
                           download= True,
                           transform= ToTensor(),
                           target_transform= None)

# %%
image,label = train_data[0]
class_names = train_data.classes
class_to_idx = train_data.class_to_idx

# %%
print(f"Image Shape: {image.shape} -> [color_channels,height,width]")
print(f"Class Label: {class_names[label]}")

plt.imshow(image.squeeze(),cmap= "gray")
plt.title(label)
# %%
BATCH_SIZE = 32

train_dataloader = DataLoader(dataset = train_data,
                              batch_size = BATCH_SIZE,
                              shuffle=True)

test_dataloader = DataLoader(dataset = test_data,
                             batch_size=BATCH_SIZE,
                             shuffle=False)

# %%
print(f"lengt of train dataloader: {len(train_dataloader)} of {BATCH_SIZE} batch size")
print(f"lengt of test dataloader: {len(test_dataloader)} of {BATCH_SIZE} batch size")

train_features_batch, train_labels_batch = next(iter(train_dataloader))
train_features_batch.shape,train_labels_batch.shape

# %%
# Tiny VGG Model
class MNISTModel(nn.Module):
    def __init__(self,input_shape:int,hidden_units:int,output_shape:int):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels= input_shape,
                      out_channels = hidden_units,
                      kernel_size = (3,3),
                      stride = 1,
                      padding = 1
                      ),
            nn.ReLU(),
            nn.Conv2d(in_channels= hidden_units,
                      out_channels = hidden_units,
                      kernel_size = (3,3),
                      stride = 1,
                      padding = 1
                      ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= 2)
            )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels= hidden_units,
                  out_channels = hidden_units,
                  kernel_size = (3,3),
                  stride = 1,
                  padding = 1
                  ),
            nn.ReLU(),
            nn.Conv2d(in_channels= hidden_units,
                      out_channels = hidden_units,
                      kernel_size = (3,3),
                      stride = 1,
                      padding = 1
                      ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features= hidden_units*7*7,
                      out_features= output_shape))
    
    def forward(self,x):
        x = self.conv_block1(x)
        #print(f"Shape of conv_block1 = {x.shape}")
        x = self.conv_block2(x)
        #print(f"Shape of conv_block2 = {x.shape}")
        x = self.classifier(x)
        #print(f"Shape of classifier= {x.shape}")
        return x

# %%
loss_fn = nn.CrossEntropyLoss()

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() 
    acc = (correct / len(y_pred)) * 100
    return acc

# %%
def train_step(model: nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device):
    
    train_loss, train_acc = 0,0
    
    model.train()
    for batch, (x,y) in enumerate(data_loader):
        x,y = x.to(device),y.to(device)
        
        y_pred = model(x)
        
        loss = loss_fn(y_pred,y)
        train_loss += loss
        
        train_acc += accuracy_fn(y_true = y,
                          y_pred = y_pred.argmax(dim=1))
        
        optimizer.zero_grad()
        
        loss.backward()
        
        optimizer.step()
        
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%\n")

# %%
def test_step(model: nn.Module,
             data_loader: torch.utils.data.DataLoader,
             loss_fn: nn.Module,
             accuracy_fn,
             device: torch.device):
    
    test_loss,test_acc = 0,0
    model.eval()
    with torch.inference_mode():
        for x,y in data_loader:
            
            x,y = x.to(device),y.to(device)
        
            test_pred = model(x)
        
            loss = loss_fn(test_pred,y)
            test_loss += loss

            test_acc += accuracy_fn(y_true = y,
                                    y_pred = test_pred.argmax(dim=1))
        
    test_acc = test_acc / len(data_loader)
    test_loss = test_loss / len(data_loader)
    
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%\n")

# %%
def eval_model(model:nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn:nn.Module,
               accuracy_fn,
               device: torch.device):
    
    loss,acc = 0,0
    model.eval()
    with torch.inference_mode():
        for x,y in tqdm(data_loader):
            x,y =x.to(device),y.to(device)
            
            y_pred = model(x)
            
            loss += loss_fn(y_pred,y)
            acc += accuracy_fn(y_true = y,y_pred = y_pred.argmax(dim=1))
            
        loss = loss / len(data_loader)
        acc = acc / len(data_loader)

    return {"model_name": model.__class__.__name__,
             "model_loss": loss.item(),
             "model_acc":acc}
# %%
def save_model(model:nn.Module,
               target_dir:str,
               model_name:str):
    
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents= True,exist_ok= True)
    
    model_save_path = target_dir_path / model_name
    
    print(f"Model is saved: {model_save_path}")
    torch.save(obj=model.state_dict(),f = model_save_path)


# %%
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device is: {device}")
    
    torch.manual_seed(7)
    model_1 =MNISTModel(input_shape= 1,
                        hidden_units = 32,
                        output_shape = len(class_names)).to(device)

    optimizer = torch.optim.SGD(params = model_1.parameters(),lr = 0.1)
    
    epochs = 3

    for epoch in tqdm(range(epochs)):
        print(f"Epoch: {epoch}\n ")
        train_step(model=model_1,
                   data_loader = train_dataloader,
                   loss_fn=loss_fn,
                   optimizer=optimizer,
                   accuracy_fn=accuracy_fn,
                   device=device)
        
        test_step(model=model_1,
                  data_loader= test_dataloader,
                  loss_fn=loss_fn,
                  accuracy_fn=accuracy_fn,
                  device = device)
    
    model_1_results = eval_model(model= model_1,
                                 data_loader= test_dataloader,
                                 loss_fn = loss_fn,
                                 accuracy_fn= accuracy_fn,
                                 device = device)

    print(model_1_results)
    
    y_preds = []
    model_1.eval()
    with torch.inference_mode():
        for x,y in tqdm(test_dataloader):
            x,y = x.to(device), y.to(device)
            
            y_logits = model_1(x)
            
            y_pred = torch.argmax(y_logits,dim=1)
            
            y_preds.append(y_pred.cpu())

    y_pred_tensor = torch.cat(y_preds)

    confmat = ConfusionMatrix(task = "multiclass",num_classes = len(class_names))
    confmat_tensor = confmat(preds = y_pred_tensor,
                             target = test_data.targets)

    fig,ax = plot_confusion_matrix(conf_mat= confmat_tensor.numpy(),
                                   class_names= class_names,
                                   figsize=(10,7))
    plt.show(fig)
    
    save_model(model= model_1, target_dir ="models", model_name= "mnist_tinyvgg_model.pth")
#%%
    if not Path("results").exists():
        Path("results").mkdir()

    fig.savefig("results/confusion_matrix.png")

# %%



