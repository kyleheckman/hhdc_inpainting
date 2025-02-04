import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import argrelextrema

def display(raw, diff1, true, masked, vmax):
    fig, axes = plt.subplots(1,4)
    
    axes[0].imshow(raw, cmap='jet', vmax=vmax)
    axes[0].axis('off')

    axes[1].imshow(diff1, cmap='jet')
    axes[1].axis('off')

    axes[2].imshow(true, cmap='jet', vmax=vmax)
    axes[2].axis('off')

    axes[3].imshow(masked, cmap='jet', vmax=vmax)
    axes[3].axis('off')

    plt.show()

def rh_vals(waveform, resolution):
    #waveform[waveform < 0.15] = 0
    if np.sum(waveform) == 0:
        return None, None
    
    energy_int = np.array(np.cumsum(waveform)/np.sum(waveform))

    energy_marks = [0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.98]
    #ground = argrelextrema(waveform, np.greater)[0][0]

    indx = [np.argwhere(energy_int>=i).min() for i in energy_marks]
    height_adjusted = (indx[1:]-indx[0])*resolution
    return indx, height_adjusted

def calc_agbd(rh, type):
	if rh is None:
		return 0

	if type == 'ent_na':
		agbd = 1.013 * np.square(-114.355 + (8.401 *  np.sqrt(rh[3] + 100)) + (3.346 *  np.sqrt(rh[-1] + 100)))
		return agbd
	elif type == 'dbt_na':
		agbd = 1.052 * np.square(-120.777 + (5.508 *  np.sqrt(rh[1] + 100)) + (3.955 *  np.sqrt(rh[-1] + 100)))
		return agbd

if __name__ == '__main__':
    data_dir = './samples'
    entries = os.listdir(data_dir)

    entries = [en for en in entries if '.npy' in en]
    print(entries)
    for en in entries:
        images = np.load(f'{data_dir}/{en}')
        images = np.transpose(images, (0,1,3,2))

        z_raw, x, x_masked = images[5], images[7], images[8]

        raw_agdb_pred = np.zeros(z_raw.shape[:2])
        agbd_true = np.zeros(x.shape[:2])
        agbd_masked = np.zeros(x_masked.shape[:2])
        

        for ix in range(z_raw.shape[0]):
              for iy in range(z_raw.shape[1]):
                    _, rh = rh_vals(z_raw[ix,iy], 1)
                    raw_agdb_pred[ix, iy] = calc_agbd(rh, type='dbt_na')
        
        for ix in range(x.shape[0]):
              for iy in range(x.shape[1]):
                    _, rh = rh_vals(x[ix,iy], 1)
                    agbd_true[ix, iy] = calc_agbd(rh, type='dbt_na')
        
        for ix in range(x_masked.shape[0]):
              for iy in range(x_masked.shape[1]):
                    _, rh = rh_vals(x_masked[ix,iy], 1)
                    agbd_masked[ix, iy] = calc_agbd(rh, type='dbt_na')

        diff1 = agbd_true-raw_agdb_pred

        agbd_max = 0
        agbd_max = np.max(raw_agdb_pred) if np.max(raw_agdb_pred) > agbd_max else agbd_max
        agbd_max = np.max(agbd_true) if np.max(agbd_true) > agbd_max else agbd_max
        agbd_max = np.max(agbd_masked) if np.max(agbd_masked) > agbd_max else agbd_max

        display(raw_agdb_pred, diff1, agbd_true, agbd_masked, agbd_max)