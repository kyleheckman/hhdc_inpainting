import numpy as np
import cv2
from random import seed, randint, choices
import string
import os
import matplotlib.pyplot as plt

class MaskGenerator():
    def __init__(
            self,
            height: int,
            width: int,
            channels: int = 1,
            rand_seed = None,
            path: str = None
    ):
        self.height = height
        self.width = width
        self.channels = channels
        self.path = path
        self.files = os.listdir(self.path) if self.path else None

        if rand_seed:
            seed(rand_seed)
    
    def _generate_mask(self):
        mat = np.zeros((self.height, self.width, self.channels), dtype=np.uint8)

        size = int((self.height + self.width) * 0.1)

        for _ in range(randint(1, 10)):
            x1, x2 = randint(1, self.width), randint(1, self.width)
            y1, y2 = randint(1, self.height), randint(1, self.height)
            thickness = randint(3, size)
            cv2.line(mat, (x1, y1), (x2, y2), (1,1,1), thickness)
        
        for _ in range(randint(1, 10)):
            x1, y1 = randint(1, self.width), randint(1, self.height)
            radius = randint(3, size)
            cv2.circle(mat, (x1, y1), radius, (1,1,1), -1)
        
        for _ in range(randint(1, 10)):
            x1, y1 = randint(1, self.width), randint(1, self.height)
            s1, s2 = randint(1, self.width), randint(1, self.height)
            a1, a2, a3 = randint(3, 180), randint(3, 180), randint(3, 180)
            thickness = randint(3, size)
            cv2.ellipse(mat, (x1,y1), (s1,s2), a1, a2, a3,(1,1,1), thickness)
        
        return 1-mat

    def _load_mask(self):
        if self.files and len(self.files) > 0:
            mask = np.load(f'{self.path}/{choices(self.files, k=1)}')

        else:
            mask_ratio = 0
            while mask_ratio < 0.25 or mask_ratio > 0.75:
                mask = self._generate_mask()
                mask = np.reshape(mask, (32,32))
                mask_ratio = np.sum(mask) / 1024
        return mask

if __name__ == '__main__':
    output_path = '../masks'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    gen = MaskGenerator(32, 32)
    saved = 0
    while saved < 128:
        mask = gen._generate_mask()

        mask = np.reshape(mask, (32, 32))
        mask_ratio = np.sum(mask) / 1024
        print(np.sum(mask))
        # Keep mask ratio between 10%-25% missing pixels
        if mask_ratio < 0.25 or mask_ratio > 0.75:
            continue

        fn = ''.join(choices(string.ascii_letters + string.digits, k=24)) + '.npy'
        np.save(f'{output_path}/{fn}', mask)
        saved += 1