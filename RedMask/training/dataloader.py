import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.utils import Sequence, to_categorical

class DataLoader(Sequence):
    def __init__(self, path_im, labels, batch_size=8, dim=(224, 224),
                 n_classes=5, n_channels=3, transforms=None, shuffle=True):
        self.dim = dim
        self.bs = batch_size
        self.labels = labels
        self.path_im = path_im
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.transforms = transforms
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.path_im) / self.bs))

    def __getitem__(self, index):
        X = np.empty((self.bs, *self.dim, self.n_channels))
        y = np.empty((self.bs), dtype=int)

        temp = self.path_im[index*self.bs:(index+1)*self.bs]
        for i, path in enumerate(temp):
            image = plt.imread(path)
            if self.transforms:
                image = self.transforms(image=image)['image']

            image = image / 255.
            X[i,] = image
            y[i] = self.labels[path]

        return X, to_categorical(y, num_classes=self.n_classes)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.path_im)
