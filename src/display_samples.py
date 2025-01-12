import numpy as np
import matplotlib.pyplot as plt
import os

def display_2d(x, x_hat, mask):
    for i in range(x.shape[0]):
        print(f'Layer {i} | MAE: {np.mean(np.abs(x[i] - x_hat[i])):.12f}')
        fig, axes = plt.subplots(2,2)
        axes[0][0].imshow(x[i], vmin=0, vmax=np.max(x), cmap='jet')
        axes[0][1].imshow(x_hat[i], vmin=0, vmax=np.max(x), cmap='jet')
        axes[1][0].imshow(mask, vmin=0, vmax=1)

        axes[0][0].axis('off')
        axes[0][1].axis('off')
        axes[1][0].axis('off')
        axes[1][1].axis('off')

        plt.show()
        plt.close()

def get_pairs(fn_list):
    pairs = []
    for i in range(0, len(fn_list)//2):
        x = None
        x_hat = None
        mask = None
        for fn in fn_list:
            x = fn if str(i) in fn and 'original' in fn else x
            x_hat = fn if str(i) in fn and 'reconstructed' in fn else x_hat
            mask = fn if str(i) in fn and 'mask' in fn else mask
        pairs.append((x, x_hat, mask))
    return pairs

def overall_error(x, xhat, mask):
    total_err = np.mean(np.abs(x - xhat))
    valid_err = np.mean(np.abs((x*mask)-(xhat*mask)))
    filled_err = np.mean(np.abs(((1-mask)*x)-((1-mask)*xhat)))

    print(f'Total MAE {total_err:.6f}')
    print(f'Valid MAE {valid_err:.6f}')
    print(f'Filled MAE {filled_err:.6f}')

if __name__ == '__main__':
    data_path = '../samples'
    fn_list = os.listdir(data_path)

    pairs = get_pairs(fn_list)
    print(pairs)
    for pair in pairs[:]:
        print(pair)
        x = np.load(f'{data_path}/{pair[0]}')
        xhat = np.load(f'{data_path}/{pair[1]}')
        mask = np.load(f'{data_path}/{pair[2]}')
        
        overall_error(x, xhat, mask)

        display_2d(x, xhat, mask)