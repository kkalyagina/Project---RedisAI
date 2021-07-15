import os
from glob import glob

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
from redisai import Client

# connecting to Redis server
con = Client(host='viper', port=6379)

# reading model and setting it to redisai
mask = open('data/frozen_model_1.15.0.pb', 'rb').read()
con.modelset("mask", 'TF', 'CPU', mask, inputs=['input_1'], outputs=['dense/BiasAdd'])

# hardcoded output translation :)
mask_state = {0: 'Properly_masked', 1: 'No_mask', 2: 'Mouth_chin', 3: 'Chin', 4: 'Nose_Mouth (no chin)'}
# hardcoded images folder :)
folder_path = 'data/images'

aug = A.Compose([
    A.Resize(224, 224, p=1),
    A.RandomBrightnessContrast(p=0.2),
    A.HorizontalFlip(p=0.5)
])


# images prepare func  todo: rewrite as generator
def transform_images(folder_path, alt_func):
    images = list()
    images_names = []
    for filepath in glob(os.path.join(folder_path, '*')):
        image = plt.imread(filepath)
        image = alt_func(image=image)['image']
        image = image / 255.
        image = np.float32(image)
        images.append(image)
        images_names.append(filepath[6:])
    return np.array(images), images_names


# transforming images
images, images_names = transform_images(folder_path, aug)

con.tensorset('img', images, dtype='float32')
con.modelrun('mask', ['img'], ['out'])
output = con.tensorget('out')
for i in range(len(output)):
    idb = np.where(output[i] == output[i].max())[0][0]
    print(images_names[i], ' - ', mask_state[idb])
