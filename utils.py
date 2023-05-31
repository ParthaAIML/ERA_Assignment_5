import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

class download_data:
    def __init__(self,dict_):
      self.dict_= dict_

    def get_data(self):
        train_transforms = transforms.Compose([
            transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),
            transforms.Resize((28, 28)),
            transforms.RandomRotation((-15., 15.), fill=0),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        # Test data transformations
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_data = datasets.MNIST('../data', train=True, download=True, transform=train_transforms)
        test_data = datasets.MNIST('../data', train=False, download=True, transform=test_transforms)

        return train_data, test_data

    def create_loader(self):
        batch_size=self.dict_["batch_size"]
        shuffle=self.dict_["shuffle"]
        num_workers=self.dict_["num_workers"]
        pin_memory=self.dict_["pin_memory"]
        kwargs = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 2, 'pin_memory': True}
        train_data, test_data=self.get_data()
        train_loader = torch.utils.data.DataLoader(train_data, **kwargs)
        test_loader = torch.utils.data.DataLoader(test_data, **kwargs)

        return train_loader,test_loader


class create_plot:

    def __init__(self,data_loader):
      self.data_loader=data_loader

    def plot_image_labels(self):
      batch_data, batch_label = next(iter(self.data_loader)) 

      fig = plt.figure()

      for i in range(12):
        plt.subplot(3,4,i+1)
        plt.tight_layout()
        plt.imshow(batch_data[i].squeeze(0), cmap='gray')
        plt.title(batch_label[i].item())
        plt.xticks([])
        plt.yticks([])

class create_accuracy_loss_plot:

    def __init__(self,train_losses,train_acc,test_losses,test_acc):
      self.train_losses=train_losses
      self.train_acc=train_acc
      self.test_losses=test_losses
      self.test_acc=test_acc

    def plot_accuray_loss(self):

      fig, axs = plt.subplots(2,2,figsize=(6,8))
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






