import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os, sys

#Image preparing function
def prepare_image(filepath, alt_func):
    image = alt_func(image=plt.imread(filepath))['image'] 
    return np.float32(image / 255.)

#Funcrion for making tensors list
def make_tensors_list(folder_path, alt_func):
    images = list()
    images_names = list()
    for filepath in glob(os.path.join(folder_path,'*')):
        images.append(prepare_image(filepath,alt_func))
        images_names.append(filepath.split('/')[-1])
    return images, images_names
