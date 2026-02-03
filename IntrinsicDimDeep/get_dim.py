from __future__ import print_function, division
import numpy as np

import time

import os
import os.path as path
from os import listdir 
from os.path import isfile, join
import pickle
from tqdm import tqdm


import torch
import torch.nn as nn
import torch.utils.data as data

# random generators init
torch.backends.cudnn.deterministic = True
torch.manual_seed(999)

# path
cwd = os.getcwd()
parts = cwd.split('/scripts/pretrained')
ROOT = parts[0]
os.chdir(ROOT)
import sys
sys.path.insert(0, ROOT)


from .IDNN.intrinsic_dimension import estimate, block_analysis
from scipy.spatial.distance import pdist, squareform



def get_intrinsic_dim(model, input_dataloader, nsamples, bs, divs, res, call_model_fn=None):
    """
    Unified function for both MLPs and Transformers.
    Works by finding the second-to-last Linear layer (last hidden layer before output).
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    def get_last_hidden_linear_layer(model: nn.Module):
        # Get all modules, checking for Linear and custom readout types
        all_modules = []
        
        for name, module in model.named_modules():
            # Check if it's a Linear layer OR a custom readout layer
            is_linear = isinstance(module, nn.Linear)
            is_readout = 'readout' in module.__class__.__name__.lower() or \
                        'mureadout' in module.__class__.__name__.lower()
            
            if is_linear or is_readout:
                all_modules.append((name, module, is_readout))
        
        if len(all_modules) < 2:
            raise ValueError(f"Model must have at least 2 linear/readout layers (hidden + output). Found {len(all_modules)}")
        
        # Print found layers for debugging
        print(f"\nFound {len(all_modules)} linear/readout layers:")
        for i, (name, layer, is_readout_flag) in enumerate(all_modules[-10:]):  # Show last 10
            layer_type = "READOUT" if is_readout_flag else "Linear"
            print(f"  [{i-len(all_modules)}] {name}: {layer.__class__.__name__} [{layer_type}]")
        
        # Find the last readout layer if it exists, otherwise use last linear
        readout_layers = [(i, name, layer) for i, (name, layer, is_readout) in enumerate(all_modules) if is_readout]
        
        if readout_layers:
            # If we have readout layers, find the Linear layer right before the last readout
            last_readout_idx, last_readout_name, last_readout = readout_layers[-1]
            
            if last_readout_idx > 0:
                # Get the layer before the readout
                last_hidden_name, last_hidden_layer, _ = all_modules[last_readout_idx - 1]
                print(f"\nFound readout layer at index {last_readout_idx}: {last_readout_name}")
                print(f"Using layer before readout: {last_hidden_name} ({last_hidden_layer.__class__.__name__})")
            else:
                raise ValueError("Readout layer is first in the model, cannot find hidden layer before it")
        else:
            # No readout layers, use second-to-last Linear (original behavior)
            last_hidden_name, last_hidden_layer, _ = all_modules[-2]
            print(f"\nNo readout layers found. Using second-to-last Linear: {last_hidden_name}")
        
        return last_hidden_layer
    
    # Get the last hidden layer 
    last_hidden_layer = get_last_hidden_linear_layer(model)
    # Register the hook
    last_hidden_layer.register_forward_hook(get_activation('last_hidden'))
    model.eval()

    ID = []
    for r in tqdm(range(divs), desc="Division"):
        Out = None
        sample_count = 0
        
        for i, data in enumerate(input_dataloader): 
            if sample_count >= nsamples:
                break
                
            # Extract representation
            for idataset, data_onedataset in enumerate(data):
                if sample_count >= nsamples:
                    break
                    
                inputs, _ = data_onedataset
                current_bs = inputs.shape[0]
                
                with torch.no_grad():
                    if call_model_fn is not None:
                        _ = call_model_fn(inputs.to(device), idataset)
                    else:
                        _ = model(inputs.to(device))
                    out = activation['last_hidden']
                
                # Flatten the output (handles both [batch, hidden] and [batch, seq, hidden])
                out_flat = out.view(current_bs, -1).cpu().data
                
                if Out is None:
                    Out = out_flat
                else:
                    Out = torch.cat((Out, out_flat), 0)
                    Out = Out.detach()
                
                sample_count += current_bs
                del out, out_flat
            
        # Compute ID
        print('Computing ID...')
        Out = Out.numpy().astype(np.float64)      
        nimgs = int(np.floor(min(nsamples, Out.shape[0]) * 0.9))
        Id = []
        
        for r in tqdm(range(res), desc="Repetition"): 
            perm = np.random.permutation(Out.shape[0])[:nimgs]        
            dist = squareform(pdist(Out[perm, :]))
            try:
                est = estimate(dist, verbose=False) 
                est = [est[2], est[3]]
            except Exception as e:
                print(f"Estimation failed: {e}")
                est = []                             
            Id.append(est)
        
        Id = np.asarray(Id)
        ID.append(Id) 
    
    print('Done.')
    
    ID = np.array(ID)
    # Filter out empty estimates
    ID_filtered = [id_vals for id_vals in ID if len(id_vals) > 0 and len(id_vals[0]) > 0]
    
    if len(ID_filtered) == 0:
        raise ValueError("All ID estimations failed!")
    
    ID_filtered = np.array(ID_filtered)
    IDs = ID_filtered[:, :, 0]
    ID_mean = np.mean(IDs)
    ID_std = np.std(IDs)
    
    return ID_mean, ID_std    

# # Toy dataset: random inputs and labels
# class ToyDataset(data.Dataset):
#     def __init__(self, n_samples=5000, input_dim=20, n_classes=3):
#         super().__init__()
#         self.x = torch.randn(n_samples, input_dim)
#         self.y = torch.randint(0, n_classes, (n_samples,))
#     def __len__(self):
#         return len(self.x)
#     def __getitem__(self, idx):
#         return self.x[idx], self.y[idx]

# # Simple feedforward network with 2 Linear layers
# class SimpleMLP(nn.Module):
#     def __init__(self, input_dim=20, hidden_dim=10, n_hidd_layers=3, output_dim=3):
#         super().__init__()
#         self.input_layer = nn.Linear(input_dim, hidden_dim)
#         self.gelu = nn.GELU()
#         layers = []
#         layers.append(self.input_layer)
#         layers.append(self.gelu)
#         for _ in range(n_hidd_layers - 1):
#             layers.append(nn.Linear(hidden_dim, hidden_dim))
#             layers.append(nn.GELU())
#         self.output_layer = nn.Linear(hidden_dim, output_dim)
#         layers.append(self.output_layer)
#         self.mlp = nn.Sequential(*layers)
#     def forward(self, x):
#         return self.mlp(x)

# import torch.optim as optim
# import torch.nn.functional as F

# def train_simple_mlp(model, dataloader, epochs=10, lr=0.01, device=None):
#     if device is None:
#         device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model = model.to(device)
#     optimizer = optim.Adam(model.parameters(), lr=lr)
#     criterion = nn.CrossEntropyLoss()
#     model.train()

#     for epoch in range(epochs):
#         running_loss = 0.0
#         correct = 0
#         total = 0

#         for inputs, labels in dataloader:
#             inputs = inputs.to(device)
#             labels = labels.to(device)

#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item() * inputs.size(0)
#             _, predicted = outputs.max(1)
#             total += labels.size(0)
#             correct += predicted.eq(labels).sum().item()

#         epoch_loss = running_loss / total
#         accuracy = 100.0 * correct / total
#         print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")

#     print("Training complete.")
#     return model


# # Parameters
# nsamples = 1000
# bs = 64
# divs = 2
# res = 3

# # Create dataset and dataloader
# dataset = ToyDataset(n_samples=5000, input_dim=5, n_classes=3)
# dataloader = data.DataLoader(dataset, batch_size=bs, shuffle=True)

# # Instantiate model
# model = SimpleMLP(input_dim=5, hidden_dim=10, n_hidd_layers=2, output_dim=3)
# trained_model = train_simple_mlp(model, dataloader, epochs=1, lr=0.01)
# # Run intrinsic dimension extraction
# ID = get_intrinsic_dim(model, dataloader, nsamples, bs, divs, res)

# # get average dimension
# IDs = ID[:,:,0]
# ID_mean = np.mean(IDs)
# ID_std = np.std(IDs)
# print("\nIntrinsic Dimension estimates shape: ", ID.shape)
# print(ID)
# print("Intrinsic Dimension: ", ID_mean, "±", ID_std)

    
    
