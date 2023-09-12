"""
Neural Network for FMT Reconstruction
- Main script for training, validation, and applying the chosen model

Created by Fay Wang
Contact: fay.wang@columbia.edu

"""

#=================#
# Import packages #
#=================#

import time
import numpy as np
import pandas as pd
import torch
import main_functions as main
import models as models
import scipy.io

#========================#
# Set up some parameters #
#========================#

# Where are we doing the training?
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# What does the mesh structure look like?
N_samples = 7000
N_measurements = 4096
N_x = 41
N_y = 41
N_z = 5

#========================#
# Create the FMT dataset #
#========================#

dataset = main.randomDataset(N_samples,N_measurements,N_x*N_y*N_z)

#===========================#
# Prepare for the main loop #
#===========================#

# What do the models look like?
n_blocks = 3
filters = 8
input_size = N_measurements
output_size = N_x*N_y*N_z

# Set hyperparameters for training and model
model_name = 'unet'
split = 0.8
learning_rate = 0.001
batch_size = 20
max_epochs = 500
patience = 50

# Set up storage tensors
TRUTHS      = torch.zeros((N_samples, N_x*N_y*N_z))
PREDICTIONS = torch.zeros((N_samples, N_x*N_y*N_z))
LOSS_TR     = torch.zeros((max_epochs))
LOSS_VA     = torch.zeros((max_epochs))

#=================#
# Start main loop #
#=================#

start_time = time.time()

# Initialize a model and move to device
model = models.UNet(in_channels=1, out_channels=1, n_blocks=n_blocks,
                    start_filters=filters,activation='relu', normalization='batch',
                    conv_mode='same', dim=3, input_size=input_size, 
                    output_size=output_size).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model using the dataset
model, LOSS_TR, LOSS_VA = main.trainModel(model, dataset, optimizer, device, split, batch_size, max_epochs, patience, start_time)

# Apply the model to the whole dataset
PREDICTIONS = main.applyModel(model, dataset, device)

# Move the model and data back to cpu from device
model = model.to('cpu')

# Finally, save the model
save_name = 'models/' + model_name + '.pth'
torch.save(model.state_dict(), save_name)
print("Saved model as:", save_name)
total_time = time.time() - start_time    

#================#
# Save the info  #
#================#

save_name = 'data/' + model_name + '_training_output.mat'
save_data = {
   'model_name'  : model_name,
   'n_blocks'    : n_blocks,
   'filters'     : filters,
   'TRUTHS'      : TRUTHS.numpy(),
   'PREDICTIONS' : PREDICTIONS.numpy(),
   'LOSS_TR'     : LOSS_TR.numpy(),
   'LOSS_VA'     : LOSS_VA.numpy(),
   'total_time'  : total_time
}

scipy.io.savemat(save_name, save_data)