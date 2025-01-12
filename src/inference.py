import torch
from torch.utils.data import DataLoader
from hhdc_vae import Autoencoder
from hhdc_data import HHDCDataset
from utils import MaskGenerator
from tqdm import tqdm
import numpy as np
import argparse
import os

def inference(
    data_path: str,
    output_path: str,
    checkpoint_path: str,
    samples: int = 1
):
    if not os.path.exists(data_path):
        print('Data folder could not be found')
        return None
    else:
        print(f'Using dataset {data_path}')
    
    test_dataset = HHDCDataset(data_path)
    print(f'Number of samples: {test_dataset.__len__()}')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'With device: {device}')

    if not os.path.exists(checkpoint_path):
        print(f'Model weights could not be found')
        return None
    
    model = Autoencoder().to(device).eval()
    model.load_state_dict(torch.load(f'{checkpoint_path}/model_weights.pt', weights_only=True))

    maskgen = MaskGenerator(32, 32)

    for i in range(samples):
        for _, x in enumerate(tqdm(test_loader, desc=f'Samples {i+1}/{samples}')):
            with torch.no_grad():
                x = x.view(-1, 32, 32).unsqueeze(1).to(device)
                mask = torch.from_numpy(maskgen._load_mask()).unsqueeze(0).unsqueeze(0).to(device).to(torch.float)

                xhat, _, _ = model(x, mask)

                x = x.view(1, 128, 32, 32).squeeze(0).cpu().numpy()
                xhat = xhat.view(1, 128, 32, 32).squeeze(0).cpu().numpy()
                mask = mask.squeeze(0).squeeze(0).cpu().numpy()

                fn = f'{output_path}/sample_{i}_original.npy'
                np.save(fn, x)

                fn = f'{output_path}/sample_{i}_reconstructed.npy'
                np.save(fn, xhat)

                fn = f'{output_path}/sample_{i}_mask.npy'
                np.save(fn, mask)
    return samples

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, help='Test dataset directory', required=True)
    
    data_path = parser.parse_args().data
    
    output_path = '../model/output'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    checkpoint_path = '../model/checkpoint'

    inference(data_path=data_path, checkpoint_path=checkpoint_path, output_path=output_path, samples=5)
