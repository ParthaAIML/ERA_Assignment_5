
# ERA Assignment 5
![era_v1](https://github.com/ParthaAIML/ERA_Assignment_5/assets/100613266/9299254f-6c29-4cca-8243-e618c7eef760)
---

## Table of contents
* [General info](#general-info)
* [Setup](#setup)
* [Files info](#files-ino)
* [Execution info](#execution-info)

## General info
`This repository contains the files used for the assignment 5 of the ERA course. The objective of this assignment is to modularize the code and store in GitHub`

## Setup
The below requirements needs to be installed before running the code
### Requirements
* `torch`
* `torchvision`
* `matplotlib`
* `torchsummary`

## Files info
There are three files iin this repository. The names of these files are below
* `model.py`
* `utils.py`
* `S5.ipynb`

The `model.py` file contains the convolution neural network model designed in a specific architecture. It contains a class called Net which has the required code for runnig the convolution neural network model.

The `utils.py` file contains all the utility functions. there are four classes in this file

 1. `download_data`
 2. `create_plot`
 3. `create_accuracy_loss_plot`
 4. `generate_model_parameters`
 
 * The `download_data` class download the data and create the dataloader
 
 * The  `create_plot` class create the plots for the images and labels

 * The `create_accuracy_loss_plot` create the accuracy and loss plot for tarining and testing

 * The `generate_model_parameters` prints the model architecture and the total parameters in each layer

The `S5.ipynb` is a ipython notebook contains all the code the run and validate the model

 * We need to import the classed defined in the `utils.py` as below

`from utils import download_data,create_plot,create_accuracy_loss_plot,generate_model_parameters`

`from model import Net`










