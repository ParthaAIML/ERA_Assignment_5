
# ERA Assignment 5

![era](https://github.com/ParthaAIML/ERA_Assignment_5/assets/100613266/71a005f6-ce58-42c9-96f8-4d0954db54bd)
---

## Table of contents
* [General info](#general-info)
* [Setup](#setup)
* [Files info](#files-ino)
* [Execution info](#execution-info)
* [Sample output](#sample-output)

### General info
`This repository contains the files used for the assignment 5 of the ERA course. The objective of this assignment is to modularize the code and store in GitHub`

### Setup
The below requirements needs to be installed before running the code

#### Requirements
* `torch`
* `torchvision`
* `matplotlib`
* `torchsummary`

### Files info
There are three files in this repository. The names of these files are 
* `model.py`
* `utils.py`
* `S5.ipynb`

The `model.py` file contains the CNN model designed in a specific architecture. It contains a class called `Net` which has the required code for runnig the CNN model.

The `utils.py` file contains all the utility functions. there are four classes in this file

 1. `download_data`
 2. `create_plot`
 3. `create_accuracy_loss_plot`
 4. `generate_model_parameters`
 
 * The `download_data` class download the train and test data `(MNIST)` and create the train dataloader and test dataloader
 
 * The  `create_plot` class create the plots for the images and labels of the train data in a `(3x4)` grid

 * The `create_accuracy_loss_plot` create the accuracy and loss plot for tarining and testing

 * The `generate_model_parameters` prints the model architecture and the total parameters in each layer

The `S5.ipynb` is a ipython notebook contains all the code to run and validate the model, create plots etc.

 * We need to import the classed defined in the `utils.py` and `model.py` as below

`from utils import download_data,create_plot,create_accuracy_loss_plot,generate_model_parameters`

`from model import Net`

### Execution info
Depending on the preference, the model can be run in google colab or in local system.The `S5.ipynb` notebook has all the code required to run the model

The repository can be cloned using the below git command

`git clone https://github.com/ParthaAIML/ERA_Assignment_5`

The uesr needs to change the directoty with the below code before importing the functionalities from `model.py` and `utils.py`

`%cd /content/gdrive/My\Drive/Assignment_5` 

### Sample output

* Plot of image and labels

![image](https://github.com/ParthaAIML/ERA_Assignment_5/assets/100613266/92c8e7ea-4bdd-404d-adf7-53dc589c88c7)


* Model Summary

![model_params](https://github.com/ParthaAIML/ERA_Assignment_5/assets/100613266/496e096d-7e42-4000-b74d-251f6f25399e)






