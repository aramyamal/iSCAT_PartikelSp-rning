from generator import generate_canada
from deeplay.applications.detection.lodestar.transforms import Transform
import torch
import numpy as np

class RandomTranslationZ(Transform):
    def __init__(self, dz=lambda: np.random.uniform(-3, 3)):
        indices = (2,)
        super().__init__(self._forward, self._backward, dz=dz, indices=indices)
    
    def _forward(self, x, dz, indices):

        images = []

        radius = lambda: np.random.uniform(45e-9, 55e-9)
        polarization_angle = lambda: np.random.rand() * 2 * np.pi
        ph= 1/16*np.pi

        image_size = 64
        for dz_one in dz:
            image = generate_canada(image_size, z = float(dz_one), radius=radius, polarization_angle=polarization_angle, ph=ph)
            image = torch.tensor(image).permute(2, 0, 1)
            image = image
            images.append(image)
        transformed = torch.stack(images).float()
        return transformed.to(x.device)
        
    @staticmethod
    def _backward(x, dz, indices):
        sub_v = torch.zeros_like(x)
        sub_v[:, indices[-1]] = dz
        return x - sub_v

class RandomScaleImage(Transform):
    def __init__(self, scale=lambda: np.random.uniform(0.8, 1.2)):
        super().__init__(self._forward, self._backward, scale=scale)

    @staticmethod
    def _forward(x, scale):
        return x * scale

    @staticmethod
    def _backward(x, scale):
        return x