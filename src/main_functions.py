"""
Neural Network for FMT Reconstruction
- Contains functions used in the main script

Created by Fay Wang
Contact: fay.wang@columbia.edu

"""

import copy
import time
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from ops import reshape_fortran, jaccardLoss

def trainModel(model, dataset, optimizer, device, split, batch_size, max_epochs, patience, start_time):
    # Overarching training function
        
    # Random split the dataset and create DataLoaders
    split = int(split*len(dataset))
    dataset_tr, dataset_va = random_split(dataset,[split, len(dataset)-split])
    loader_tr = DataLoader(dataset_tr, batch_size=batch_size)
    loader_va = DataLoader(dataset_va, batch_size=batch_size)
    
    # Prepare some variables
    loss_tr = torch.zeros((max_epochs))
    loss_va = torch.zeros((max_epochs))
    pat = patience
    loss_va_min = 10e6
    
    # Start main training loop
    for epoch in range(max_epochs):
        
        # Do training and validation
        loss_tr[epoch] = train(model, loader_tr, optimizer, device)
        loss_va[epoch] = valid(model, loader_va, device)
        
        # Print an update every once in a while
        if (epoch) % 1 == 0:
            elapsed = time.time() - start_time
            hrs = int(elapsed//3600)
            mins = int((elapsed-3600*hrs)//60)
            sec = int((elapsed-3600*hrs-60*mins)//1)
            print('# ({:02d}:{:02d}:{:02d}) Epoch {:3d} | Training Loss {:.4f} | Validation Loss {:.4f}'
                .format( hrs, mins, sec, epoch, loss_tr[epoch], loss_va[epoch] ))
                
        # Maybe stop early due to non-decreasing validation loss
        if loss_va[epoch] <= loss_va_min:
            loss_va_min = loss_va[epoch]
            best_model  = copy.deepcopy(model)
            pat = patience
            best_epoch = epoch
        else:
            pat -= 1
            if pat == 0:
                for model_w, best_w in zip(model.parameters(),best_model.parameters()):
                    model_w.data = best_w.data
                print("# Training stopped early on epoch", best_epoch, "and best weights loaded back...")
                print("#=========================================================")
                break
    
    return model, loss_tr, loss_va
    
    
def train(model, loader_tr, optimizer,device):
    model.train() # Put the model in training mode 
    jaccard = jaccardLoss() # use Jaccard loss for better image quality
    loss = 0
    
    for i, (x,y) in enumerate(loader_tr):
        optimizer.zero_grad()
        x = x.to(device)
        y = y.to(device)

        # Forward pass
        out = model(x)
        out = out[:,0,0:41,0:41,0:5] # UNet output has extra zeros, crop them out
        out = reshape_fortran(out,[x.size(0),y.size(1)])
        loss_mse = torch.nn.functional.mse_loss(out, y)
        loss_jac = jaccard(out,y)
        loss = loss_mse + loss_jac # combined loss f(x)
        loss.backward()
        optimizer.step()
    return loss

def valid(model, loader_va, device):
    model.eval() # Puts the model in eval mode
    jaccard = jaccardLoss() 
    loss = 0
    
    with torch.no_grad():
        for i, (x,y) in enumerate(loader_va):
            x = x.to(device)
            y = y.to(device)

            # Forward pass
            pred = model(x)
            pred = pred[:,0,0:41,0:41,0:5] # UNet output has extra zeros, crop them out
            pred = reshape_fortran(pred,[x.size(0),y.size(1)])
            loss_mse = torch.nn.functional.mse_loss(pred, y)
            loss_jac = jaccard(pred,y)
            loss = loss_mse + loss_jac
            
    return loss


def applyModel(model, dataset, device):
    # Apply the model to the entire dataset
    model.eval() # Put model in eval mode for this
    
    # Initialize storage for predictions
    predictions = torch.zeros((len(dataset), dataset[0][1].shape[0]))
    
    with torch.no_grad():
        for i, (x,y) in enumerate(dataset):
            x = x.to(device)
            x = torch.unsqueeze(x,0) # for "single" batch size
            pred = model(x)
            pred = pred[:,0,0:41,0:41,0:5] 

            pred = reshape_fortran(pred,[x.size(0),y.size(0)])
            
            predictions[i,:] = pred.squeeze()
    
    return predictions.to('cpu')

class CustomDataset(Dataset):
    def __init__(self, meas_path, ops_path, bkg, timeIndex):
        print("measurement set: ", meas_path, "\n")
        print("optical property set:", ops_path)
        cv1 = pd.read_csv(self.meas_path,header=None)
        cv2 = pd.read_csv(self.ops_path,header=None)
        dataX = cv1.iloc[0:,0:]
        dataY = cv2.iloc[0:,0:]
        X = dataX.values
        Y=dataY.values[:,6724:15129]
        for i in range(len(Y)):
            noise = np.random.normal(0,.05,8405)
            Y[i,:] = Y[i,:] + noise

        X = np.log(X) - np.log(bkg)
        self.dataset = TensorDataset(torch.Tensor(X[:,timeIndex]),torch.Tensor(Y))

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        datax, datay = self.dataset[idx]
        return datax, datay

def randomDataset(N_samples, N_measurements, N_nodes):
# A random dataset will be created

    # Random input measurements and outputs
    data_meas = torch.rand((N_samples, N_measurements))
    data_out   = torch.rand((N_samples, N_nodes))

    # Create dataset object
    dataset = TensorDataset(torch.Tensor(data_meas), torch.Tensor(data_out))

    # Return the dataset
    return dataset