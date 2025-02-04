import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import argparse

from ..model.diffusion_unet import DiffusionUnet
from ..utils.datasets import HHDCDataset
from ..utils.schedulers import NoiseScheduler

'''
For training V1 of DiffusionUnet
    -> only 1 ResBlock per layer
    -> self-attention enabled
    -> no T2I adapter
    -> no geo-data cross-attention
'''


def train(
        data_path: str,
        checkpoint_path: str,
        time_steps: int = 1000,
        batch_size: int = 32,
        epochs: int = 50,
        lr = 2e-4
):
    # Initialize dataset and dataloader
    train_dataset = HHDCDataset(data_path, get_slice=True)
    print(f'Dataset: {data_path} | Size: {train_dataset.__len__()}')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'With device: {device}')

    # Initialize model and optimizer
    model = DiffusionUnet(time_steps=time_steps).to(device)
    optimizer = Adam(model.parameters(), lr=lr)

    # Load checkpoint if present
    if checkpoint_path:
        print('Loading checkpoint ...')
        model.load_state_dict(torch.load(f'{checkpoint_path}/diffusion_weights.pt', weights_only=True))
        optimizer.load_state_dict(torch.load(f'{checkpoint_path}/optim_diff_weights.pt', weights_only=True))
    else:
        print('No checkpoint loaded')

    # Set loss function
    loss_function = nn.MSELoss(reduction='mean')

    # Initialize noise scheduler
    noise_sched = NoiseScheduler(num_time_steps=time_steps)

    # Training routine
    for i in range(epochs):
        epoch_loss = 0
        for _, x in enumerate(tqdm(train_loader, desc=f'Epoch {i+1}/{epochs}')):
            # Zero gradients
            optimizer.zero_grad()

            x = x.to(device)

            # Get random timestep
            t = torch.randint(0, time_steps, (x.shape[0],))

            # Sample from gaussian distribution
            eps = torch.randn_like(x, requires_grad=False)
            a = noise_sched.alpha[t].view(x.shape[0],1,1,1).to(device)

            # Noise x to input to DDPM
            x_noised = (torch.sqrt(a)*x) + (torch.sqrt(1-a)*eps)

            # Predict noise
            pred = model(x_noised, t)

            # Compute loss and backpropogate
            loss = loss_function(pred, eps)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        # Print result
        print(f'Epoch {i+1} | Loss: {epoch_loss/batch_size:.6f}')

        # Save checkpoint
        if not os.path.exists('../weights/ddpm'):
            os.makedirs('../weights/ddpm')
        torch.save(model.state_dict(), '../weights/ddpm/diffusion_weights.pt')
        torch.save(optimizer.state_dict(), '../weights/ddpm/optim_diff_weights.pt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, help='Training dataset directory', required=True)
    
    data_path = parser.parse_args().data

    if not os.path.exists('../weights'):
        os.makedirs('../weights')
    
    checkpoint_path = None
    if os.path.exists('../weights/ddpm/diffusion_weights.pt') and os.path.exists('../weights/ddpm/optim_diff_weights.pt'):
        checkpoint_path = '../weights/ddpm'
    
    train(data_path=data_path, checkpoint_path=checkpoint_path, batch_size=128, epochs=100, lr=1e-6)