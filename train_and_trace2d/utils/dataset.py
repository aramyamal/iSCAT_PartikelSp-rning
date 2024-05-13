import torch
import os
from PIL import Image
from torchvision import transforms
import numpy as np
import h5py

class Particle_dataset(torch.utils.data.Dataset):

    # Base Particle_dataset from which all specialized data sets inherit
    # Contains necessary methods for torch, such as __len__ and __getitem__, 
    # and some useful methods, such as __add__ and __setitem__

    def __init__(self):
        self.images=[]

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, i):
        return self.images[i]
    
    def __setitem__(self, key, value):
        self.images[key] = value
        
    def __add__(self, data):
        if 'dataset' in str(type(data)) or 'Matlab' in str(type(data)):
            self.images += data.images
        else:
            self.images += [data]
        return self
    
    def __neg__(self):
        for i in range(len(self.images)):
            self.images[i] = -self.images[i]
        return self
    
    def __call__(self):
        return self.images
    
    def delete(self, i):
        self.images = [image for image, j in enumerate(self.images) if j != i]

class Image_dataset(Particle_dataset):

    # Creates a Particle_dataset from all images (.png) contained in the folder path
    # Training the network on .png files is not recommended due to possible loss of information
    # during saving and loading from our method, but could work if care is taken.

    def __init__(self, path):
        super().__init__()
        self.folder_path = path
        self.extract_pngs(path)

    def extract_pngs(self, folder_path):
        self.image_paths = []
        for filename in os.listdir(folder_path):
            filepath = os.path.join(folder_path, filename)
            if os.path.isfile(filepath):
                self.image_paths.append(filepath)

        transform = transforms.ToTensor()
        for image_path in self.image_paths:
            img = Image.open(image_path)
            img_tensor = transform(img)
            self.images.append(img_tensor)

class Array_dataset(Particle_dataset):

    # Creates a Particle_dataset from all arrays (.npy) contained in the folder path

    def __init__(self, path):
        super().__init__()
        self.folder_path = path
        self.extract_arrays(path)

    def extract_arrays(self, folder_path):
        self.image_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path)]    
        
        for image_path in self.image_paths:
            image = np.load(image_path).astype('float32')
            img_tensor = torch.from_numpy(image).unsqueeze(0)
            self.images.append(img_tensor)

class Matlab(Particle_dataset):

    # Create a Particle_dataset from a matlab file with images in range "Range"
    # If Range is not specified, all images in the matlab file will be loaded which takes
    # a lot of space and time

    def __init__(self, path=None, Range=None):
        super().__init__()
        file = h5py.File(path)['Im_stack']
        if Range!=None:
            for i in range(*Range):
                image = file[i]
                img_tensor = torch.from_numpy(image).unsqueeze(0)
                self.images.append(img_tensor.type(torch.float32))
        else:
            for image in file:
                img_tensor = torch.from_numpy(image).unsqueeze(0)
                self.images.append(img_tensor.type(torch.float32))