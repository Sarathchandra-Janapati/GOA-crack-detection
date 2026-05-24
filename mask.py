import cv2
import tensorflow
import numpy as np
from tensorflow.keras.utils import Sequence
import math

class PGenerator(Sequence):

    def __init__(self, x_set, y_set, batch_size=5, img_dim=(448, 448), augment=False, gamma=1.0):
        self.x = x_set
        self.y = y_set
        self.batch_size = batch_size
        self.img_dim = img_dim
        self.augment = augment
        self.gamma = gamma

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def apply_power_law(self, image, gamma=2.0):
        normalized_image = image / 255.0
        gamma_corrected = np.array(255 * (normalized_image ** gamma), dtype='uint8')
        return gamma_corrected

    def augment_image(self, image):
        # Add your augmentation techniques here
        # For example, you can flip, rotate, or adjust brightness/contrast
        # Here's an example of flipping horizontally with 50% probability
        if np.random.rand() < 0.5:
            image = cv2.flip(image, 1)  # Flip horizontally
        # You can add more augmentation techniques as needed
        return image

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        augmented_batch_x = []
        augmented_batch_y = []

        for file_name_x, file_name_y in zip(batch_x, batch_y):
            img_x = cv2.imread(file_name_x, -1)
            img_y = cv2.imread(file_name_y, -1)

            img_x = cv2.cvtColor(img_x, cv2.COLOR_BGR2RGB)
            img_y = cv2.cvtColor(img_y, cv2.COLOR_BGR2RGB)

            img_x = cv2.resize(img_x, (self.img_dim[1], self.img_dim[0]))
            img_y = cv2.resize(img_y, (self.img_dim[1], self.img_dim[0]))

            if self.augment:
                img_x = self.augment_image(img_x)
                img_y = self.augment_image(img_y)

            img_x = self.apply_power_law(img_x, gamma=self.gamma)

            augmented_batch_x.append(img_x)
            augmented_batch_y.append(img_y)

        batch_x = np.array(augmented_batch_x, dtype='uint8')
        batch_y = np.array(augmented_batch_y, dtype='uint8')

        batch_y = np.expand_dims(batch_y, -1)
        
        return batch_x / 255, batch_y / 255
