
# ERA Assignment 5


![era](https://github.com/ParthaAIML/ERA_Assignment_5/assets/100613266/71a005f6-ce58-42c9-96f8-4d0954db54bd)
---

## Table of contents
---
* [General info](#general-info)
* [Setup](#setup)
* [Files info](#files-ino)
* [Execution info](#execution-info)
* [Sample output](#sample-output)

### General info
---
`This repository contains the files used for the assignment 5 of the ERA course. Here we have build a CNN model to predict the hand written digits [0-9] using MNIST data availabe with torchvision library. The objective of this assignment is to modularize the code and store in GitHub`

### Setup
---
The below requirements needs to be installed before running the code

#### Requirements
* `torch`
* `torchvision`
* `matplotlib`
* `torchsummary`

### Files info
---
There are three files in this repository. The names of these files are 
*  **`model.py`**
*  **`utils.py`**
*  **`S5.ipynb`**

The **`model.py`** file contains the CNN model designed in a specific architecture. It contains a class called `Net` which has the required code for runnig the CNN model.

The **`utils.py`** file contains all the utility functions. there are six utility functions in this file

 1. `GetCorrectPredCount`
 2. `train`
 3. `test`
 4. `download_data`
 5. `create_accuracy_loss_plot`
 6. `generate_model_parameters`
 
 * The `GetCorrectPredCount` function counts the correct predictions by the model.
 
 * The  `train` function trains the model
 
 * The  `test` function test the model performance

 * The  `download_data` function will download the MNIST train and test data, we need to specify the train and test transformation in the function.

 * The `create_accuracy_loss_plot` create the accuracy and loss plot for tarining and testing

 * The `generate_model_parameters` prints the model architecture and the total parameters in each layer

The **`S5.ipynb`** is a ipython notebook contains all the code to run and validate the model, create plots etc.

 * We need to import the functions defined in the `utils.py` and `model.py` as below

  `from utils import  GetCorrectPredCount,train,test,download_data,create_accuracy_loss_plot,generate_model_parameters`

  `from model import Net`
  
  The user needs to run all the cells of the S5.ipynb notebook to get the complete output

### Execution info
---
Depending on the preference, the model can be run in google colab or in local system.The `S5.ipynb` notebook has all the code required to run the model

The repository can be cloned using the below git command

`git clone https://github.com/ParthaAIML/ERA_Assignment_5`

The uesr needs to change the directoty with the below code before importing the functionalities from `model.py` and `utils.py`

`%cd /content/gdrive/My\Drive/Assignment_5` 

### Sample output
---
* Accuracy and Loss Plots

![accuracy_loss](https://github.com/ParthaAIML/ERA_Assignment_5/assets/100613266/f8330383-97db-438d-ae74-fb9431c1bfdb)

* Model Summary

![model_params](https://github.com/ParthaAIML/ERA_Assignment_5/assets/100613266/496e096d-7e42-4000-b74d-251f6f25399e)






