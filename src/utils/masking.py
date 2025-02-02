import numpy as np
from random import seed, randint
import os
import cv2

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

        for x in np.random.randint(0,self.width,randint(12,40)):
            y1, y2 = 0, self.height
            cv2.line(mat, (x, y1), (x, y2), (1,1,1), 1)

        return 1-mat