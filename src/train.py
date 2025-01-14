import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.models import vgg16
from hhdc_vae import Autoencoder
from hhdc_data import HHDCDataset
from utils import MaskGenerator
from tqdm import tqdm
import argparse
import os

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.l1 = nn.L1Loss(reduction='mean')
    
    def forward(self, perc_out, perc_comp, perc_gt):
        loss = 0
        for o, c, g in zip(perc_out, perc_comp, perc_gt):
            loss += self.l1(o, g) + self.l1(c, g)
        return loss

class TotalLoss(nn.Module):
    def __init__(self, vgg_model):
        super(TotalLoss, self).__init__()
        self.l1 = nn.L1Loss(reduction='mean')
        self.lp = PerceptualLoss()
        self.vgg = vgg_model

    def loss_valid(self, xhat, x, mask):
        # Compute L1 loss between values outside of mask
        return self.l1(mask * xhat, mask * x)
    
    def loss_hole(self, xhat, x, mask):
        # Compute L1 loss between values within the mask
        return self.l1((1-mask) * xhat, (1-mask) * x)

    def loss_kld(self, mean, logvar):
        # Compute KLD of the mean and logvar layers
        return -0.5 * torch.sum(1 + logvar - torch.square(mean)  - torch.exp(logvar))
    
    def loss_tv(self, x):
        # Compute pixelwise total variation loss of reconstructed image
        loss_x = torch.mean(torch.abs(x[:,:,1:,:] - x[:,:,:-1,:]))
        loss_y = torch.mean(torch.abs(x[:,:,:,1:] - x[:,:,:,:-1]))
        return loss_x + loss_y

    def forward(self, xhat, x, mask, mean, logvar):
        xcomp = (mask * x) + ((1-mask) * xhat)
        
        # Get output masks from VGG16 for perceptual loss calculations
        vgg_out = self.vgg.get_features(torch.cat((xhat,xhat,xhat), axis=1))
        vgg_comp = self.vgg.get_features(torch.cat((xcomp,xcomp,xcomp), axis=1))
        vgg_gt = self.vgg.get_features(torch.cat((x,x,x), axis=1))

        l1 = self.loss_valid(xhat, x, mask)
        l2 = self.loss_hole(xhat, x, mask)
        l3 = self.lp(vgg_out, vgg_comp, vgg_gt)
        l4 = self.loss_kld(mean, logvar)
        l5 = self.loss_tv(xcomp)

        ''' Compute weighted total loss, weighed by hyperparameters
        lambda_1 = 1
        lambda_2 = 6
        lambda_3 = 0.05
        lambda_4 = 1
        '''
        # print(f'L1: {l1}')
        # print(f'L2: {l2}')
        # print(f'L3: {l3}')
        # print(f'L4: {l4}')

        return l1 + (7*l2) + (0.2*l3) + (0.1*l5)# + (0.5*l4)

class VGGFeatureExtractor():
    def __init__(self, device):
        self.device = device
        self.model = vgg16(weights='IMAGENET1K_FEATURES').to(self.device).eval()
    
    def get_features(self, x):
        pool1 = self.model.features[:3](x)
        pool2 = self.model.features[:6](x)
        pool3 = self.model.features[:10](x)
        return (pool1, pool2, pool3)

def initialize_vgg(device):
    model = vgg16(weights='IMAGENET1K_FEATURES').to(device).eval()
    model_out = [model.features[:3], model.features[:6], model.features[:10]]
    return model_out

def train(
        data_path: str,
        checkpoint_path: str,
        batch_size: int = 4,
        epochs: int = 100,
        lr = 2e-4
):
    train_dataset = HHDCDataset(data_path)
    print(f'Dataset size: {train_dataset.__len__()}')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'With device: {device}')

    model = Autoencoder().to(device)
    optimizer = Adam(model.parameters(), lr=lr)

    if checkpoint_path:
        print('Loading checkpoint ...')
        model.load_state_dict(torch.load(f'{checkpoint_path}/model_weights.pt', weights_only=True))
        optimizer.load_state_dict(torch.load(f'{checkpoint_path}/optim_weights.pt', weights_only=True))
    else:
        print('No checkpoint loaded')

    with torch.no_grad():
        vgg = VGGFeatureExtractor(device)

    maskgen = MaskGenerator(32, 32)

    loss_function = TotalLoss(vgg)

    for i in range(epochs):
        epoch_loss = 0
        for _, x in enumerate(tqdm(train_loader, desc=f'Epoch {i+1}/{epochs}')):
            optimizer.zero_grad()

            x = x.view(-1, 32, 32).unsqueeze(1).to(device)
            mask = torch.from_numpy(maskgen._load_mask()).unsqueeze(0).unsqueeze(0).to(device).to(torch.float)
            
            xhat, mean, logvar = model(x, mask)
            
            loss = loss_function(xhat, x, mask, mean, logvar)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        print(f'Epoch {i+1} | Loss: {epoch_loss/batch_size:.6f}')

        if not os.path.exists('../model/checkpoint'):
            os.makedirs('../model/checkpoint')
        torch.save(model.state_dict(), '../model/checkpoint/model_weights.pt')
        torch.save(optimizer.state_dict(), '../model/checkpoint/optim_weights.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, help='Training dataset directory', required=True)
    
    data_path = parser.parse_args().data
    print(f'Training dataset: {data_path}')

    if not os.path.exists('../model'):
        os.makedirs('../model')
    
    checkpoint_path = None
    if os.path.exists('../model/checkpoint/model_weights.pt') and os.path.exists('../model/checkpoint/optim_weights.pt'):
        checkpoint_path = '../model/checkpoint'
    
    # Reset checkpoint path to None to start from scratch
    #checkpoint_path = None

    train(data_path=data_path, checkpoint_path=checkpoint_path, batch_size=1, epochs=50, lr=1e-6)