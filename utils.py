import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm

train_losses = []
test_losses = []
train_acc = []
test_acc = []

test_incorrect_pred = {'images': [], 'ground_truths': [], 'predicted_vals': []}

def GetCorrectPredCount(pPrediction, pLabels):
  return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

def train(model, device, train_loader, optimizer):
  model.train()
  pbar = tqdm(train_loader)

  train_loss = 0
  correct = 0
  processed = 0

  for batch_idx, (data, target) in enumerate(pbar):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()

    # Predict
    pred = model(data)
    # print("pred",pred)
    # print("target",target)

    # Calculate loss
    loss = F.nll_loss(pred, target)
    train_loss+=loss.item()

    # Backpropagation
    loss.backward()
    optimizer.step()
    
    correct += GetCorrectPredCount(pred, target)
    processed += len(data)

    pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

  train_acc.append(100*correct/processed)
  train_losses.append(train_loss/len(train_loader))
  return train_acc,train_losses
  


def test(model, device, test_loader):
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss

            correct += GetCorrectPredCount(output, target)


    test_loss /= len(test_loader.dataset)
    test_acc.append(100. * correct / len(test_loader.dataset))
    test_losses.append(test_loss)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_acc,test_losses

def download_data(train_transform,test_transform):
  train_data = datasets.MNIST('../data', train=True, download=True, transform=train_transform)
  test_data = datasets.MNIST('../data', train=False, download=True, transform=test_transform)
  
  return train_data,test_data

class create_accuracy_loss_plot:

    def __init__(self,train_losses,train_acc,test_losses,test_acc):
      self.train_losses=train_losses
      self.train_acc=train_acc
      self.test_losses=test_losses
      self.test_acc=test_acc

    def plot_accuray_loss(self):

      fig, axs = plt.subplots(2,2,figsize=(8,8))
      axs[0, 0].plot(self.train_losses)
      axs[0, 0].set_title("Training Loss")
      axs[1, 0].plot(self.train_acc)
      axs[1, 0].set_title("Training Accuracy")
      axs[0, 1].plot(self.test_losses)
      axs[0, 1].set_title("Test Loss")
      axs[1, 1].plot(self.test_acc)
      axs[1, 1].set_title("Test Accuracy")

class generate_model_parameters:

    def __init__(self,input_shape):
      self.input_shape=input_shape

    def generate_params(self,model_class):
      self.model_class=model_class 
      from torchsummary import summary
      use_cuda = torch.cuda.is_available()
      device = torch.device("cuda" if use_cuda else "cpu")
      model = model_class().to(device)
      summary(model, input_size=(1, self.input_shape, self.input_shape))






