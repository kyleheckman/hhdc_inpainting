import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np
import argparse

from ..model.diffusion_unet import DiffusionUnet
from ..model.t2i_adapter import T2IAdapter
from ..utils.datasets import HHDCDataset
from ..utils.schedulers import NoiseScheduler
from ..utils.masking import MaskGenerator

'''
For testing V1 of DiffusionUnet
    -> only 1 ResBlock per layer
    -> self-attention enabled
    -> no T2I adapter
    -> no geo-data cross-attention
'''

def inference(
        data_path: str,
        diff_path: str,
        adapter_path: str,
        output_path: str,
        time_steps: int = 1000,
        samples: int = 5,
        samples_len: int = 16
):
    # Check for valid data path
    if not os.path.exists(data_path):
        print('Data folder could not be found')
        return None
    else:
        print(f'Using dataset {data_path}')

    # Check for valid path to weights
    if not os.path.exists(diff_path):
        print(f'Diffusion weights could not be found')
        return None
    if not os.path.exists(adapter_path):
        print(f'Adapter weights could not be found')
        return None
    
    # Initialize dataset
    test_dataset = HHDCDataset(data_path)
    print(f'Number of samples: {test_dataset.__len__()}')

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'With device: {device}')

    # Initialize models
    diffusion = DiffusionUnet(time_steps=time_steps).to(device).eval()
    diffusion.load_state_dict(torch.load(f'{diff_path}/diffusion_weights.pt', weights_only=True))

    adapter = T2IAdapter(num_layers=4, channels=[32,64,128,256], norms=[(128,64),(64,32),(32,16),(16,8)]).to(device)
    adapter.load_state_dict(torch.load(f'{adapter_path}/adapter_weights.pt', weights_only=True))

    # Initialize noise scheduler
    noise_sched = NoiseScheduler(num_time_steps=time_steps)

    # Initialize mask generator
    maskgen = MaskGenerator(128,64)

    # Select timesteps to output
    times = [50, 100, 200, 400, 700]
    images = []

    # Inference routine
    n = 0
    with torch.no_grad():
        for i, index in enumerate(tqdm(np.random.randint(0,test_dataset.__len__(),samples), desc=f'Sample {n}/{samples}')):
            x = test_dataset.__getitem__(index).unsqueeze(0).to(device)

            # Reshape x from 3D: (B, C, H, W, D) -> 2D: (BD, C, H, W)
            x = torch.permute(x, (0,4,1,2,3))
            x = torch.reshape(x, (-1,1,128,64))
            # Get samples_len number of 2D slices from HHDC
            x = x[0:samples_len]

            # Get mask
            mask = maskgen._generate_mask()
            mask = torch.tensor(mask).squeeze(2).unsqueeze(0).unsqueeze(0).to(torch.float).to(device)
            mask = torch.cat([mask]*x.shape[0], dim=0)

            x_masked = x * mask

            # Get adapter output
            t2i_out = adapter(x_masked, [3,9,15])

            # Samples gaussian noise as initial input to diffusion
            z = torch.randn_like(x_masked)

            for t in reversed((range(1,time_steps))):
                t = [t]

                temp = (noise_sched.beta[t]/((torch.sqrt(1-noise_sched.alpha[t]))*(1/torch.sqrt(1-noise_sched.beta[t]))))
                z = (1/(torch.sqrt(1-noise_sched.beta[t].to(device))))*z - (temp.to(device)*diffusion(z,t,t2i_out))

                # Save selected intermediate samples
                if t[0] in times:
                    images.append(z.squeeze(1).unsqueeze(0).cpu())
                
                eps = torch.randn_like(x_masked)
                z = z + (eps*torch.sqrt(noise_sched.beta[t].to(device)))

                # compute x_known = sqrt(alpha_hat)*actual + (1-alpha_hat)*eps
                x_known = torch.sqrt(noise_sched.alpha[t].to(device))*x_masked + (1-noise_sched.alpha[t].to(device))*eps
                # combine x_known and x_unknown
                z = x_known*mask + (1-mask)*z
            
            temp = noise_sched.beta[0]/( (torch.sqrt(1-noise_sched.alpha[0]))*(torch.sqrt(1-noise_sched.beta[0])))
            z = (1/(torch.sqrt(1-noise_sched.beta[0])))*z - (temp*diffusion(z,[0], t2i_out))

            # Get predicted output xhat
            z = z.squeeze(1).unsqueeze(0).cpu()
            # Get masked sample
            x_masked = (x * mask).squeeze(1).unsqueeze(0).cpu()
            # Get input w/0 mask
            x = x.squeeze(1).unsqueeze(0).cpu()

            # Append to images
            images.append(z)
            images.append(x-z)
            images.append(x)
            images.append(x_masked)
            images.append(mask.squeeze(1).unsqueeze(0).cpu())

            print(F'After: {len(images)}')

            # Condense and save images
            images = torch.concat(images, axis=0).numpy()
            fn = f'{output_path}/sample_{i}_total_grouped.npy'
            np.save(fn, images)
            images = []

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, help='Training dataset directory', required=False)
    
    data_path = parser.parse_args().data
    print(f'Training dataset: {data_path}')

    output_path = '../diffusion_v1/output_t2i'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    diff_path = './weights/ddpm'
    adapter_path = './weights/t2i'

    inference(data_path=data_path, diff_path=diff_path, adapter_path=adapter_path, output_path=output_path, samples=5)