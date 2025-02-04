import numpy as np
import matplotlib.pyplot as plt
import os

def display_reverse(images):
    fig, axes = plt.subplots(1,10)
    for i, ax in enumerate(axes.flat):
        x = images[i]
        if i in [5,7,8,9]:
            ax.imshow(x, cmap='jet', aspect=0.5, interpolation='nearest', vmin=0, vmax=1)
        else:
            ax.imshow(x, cmap='jet', aspect=0.5, interpolation='nearest')
        
        ax.axis('off')
    plt.show()

if __name__ == '__main__':
    data_dir = '../samples'
    entries = os.listdir(data_dir)

    entries = [en for en in entries if '.npy' in en]
    print(entries)
    for en in entries:
        images = np.load(f'{data_dir}/{en}')
        print(images.shape)
        images = np.transpose(images, (1,0,2,3))

        for layer in range(images.shape[0]):
            print(f'Layer {layer}')
            display_reverse(images[layer])