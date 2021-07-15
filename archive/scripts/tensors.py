import albumentations as A
import datetime, random, time
import pickle as pkl

from RedMask.utils.tensors_func import make_tensors_list

#Arguments
folder_path = 'data/images'

aug = A.Compose([
    A.Resize(224, 224, p=1),
    A.RandomBrightnessContrast(p=0.2),
    A.HorizontalFlip(p=0.5)
    ])

#Making tensor list
images, images_names = make_tensors_list(folder_path, aug)

#Saving tensor list
with open('data/tensors/tensorlist.pkl','wb') as fl:
    pkl.dump(images, fl, protocol=2)

with open('data/tensors/img_names.pkl','wb') as fl:
    pkl.dump(images_names, fl, protocol=2)


