import random
from typing import NoReturn

import numpy as np
import torch
from torchvision.transforms import v2
from torchvision import transforms as v1
import random


scale = v1.Lambda(
    lambda x: x * 2 - 1
)

inverse_scale = v1.Lambda(
    lambda x: (x + 1) / 2
)


class RandomTransform:
    def __init__(self, transforms_list):
        self.transforms_list = transforms_list

    def __call__(self, x: torch.Tensor, prompt: str = None):
        x_transform, prompt = random.choice(self.transforms_list)(x, prompt)
        return x, x_transform, prompt
        

class Generation:
    def __init__(self):
        pass

    def __call__(self, x, prompt):
        return torch.randn_like(x), prompt
        

class JPEGCompression:
    def __init__(self, input_size: int = 256, quality_range: list = [20, 90]):
        self.input_size = input_size
        self.quality_range = quality_range
        
    def __call__(self, x: torch.Tensor, prompt: str = None) -> torch.Tensor:
        quality = random.randint(self.quality_range[0], self.quality_range[1])
        # Define the JPEG compression transform
        jpeg_transform = v2.JPEG(quality)
        
        # Apply the transform
        x = jpeg_transform((inverse_scale(x) * 255).to(torch.uint8)).float() / 255
        x = scale(x)

        if prompt is not None and prompt != '':
            prompt = "Restore this image"
            return x, prompt

        return x, prompt


class DownScale:
    def __init__(self, input_size: int = 256, sizes: list = [16, 32, 64, 128, 256]):
        
        self.resizes = [
            v1.Resize((size, size)) for size in sizes
        ]

        self.out_resize = v1.Resize((input_size, input_size))

    def __call__(self, x, prompt: str = None):
        resize_idx = random.randint(0, 4)
        x = self.out_resize(self.resizes[resize_idx](x))

        if prompt is not None and prompt != '':
            prompt = "Upscale this image"
            return x, prompt

        return x, prompt


class Noising:
    def __init__(self, input_size: int = 256, strength_range: list = [0, 0.5]):

        self.strengths = torch.range(
            start=strength_range[0], 
            end=strength_range[1], 
            step=(strength_range[1] - strength_range[0]) / 100
        ) 
        
        self.noise_type = "normal"
        
        self.noise_function = torch.randn_like

    def __call__(self, x, prompt: str = None):
        strength_idx = random.randint(0, 99)
        strength = self.strengths[strength_idx]

        noise = self.noise_function(x)

        x = x + torch.randn_like(noise) * strength
    
        # Clip the values to [0, 1]
        x = torch.clamp(x, -1, 1.)
    
        if prompt is not None and prompt != '':
            prompt = "Denoise this image"
            return x, prompt

        return x, prompt


class GaussinaBlur:
    def __init__(self, input_size: int = 256, kernel_size=(15, 15), sigma=(0.1, 2.0)):
        
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.sigma = sigma

        self.bluring = v1.GaussianBlur(
            kernel_size=self.kernel_size, 
            sigma=self.sigma
        )

    def __call__(self, x, prompt: str = None):
        x = self.bluring(x)
    
        # Clip the values to [0, 1]
        x = torch.clamp(x, -1., 1.)
    
        if prompt is not None and prompt != '':
            prompt = "Deblur this image"
            return x, prompt

        return x, prompt


class Grayscale:
    def __init__(self, input_size: int = 256):
        self.input_size = input_size
        self.gray = v1.Grayscale(num_output_channels=3)

    def __call__(self, x, prompt: str = None):
        x = self.gray(x)
    
        # Clip the values to [0, 1]
        x = torch.clamp(x, -1., 1.)
    
        if prompt is not None and prompt != '':
            prompt = "Color this image"
            return x, prompt

        return x, prompt