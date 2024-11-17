from PIL import Image
import torch

import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

from .view_base import BaseView


class Rotate90CWView(BaseView):
    def __init__(self):
        pass

    def view(self, im, background=None, **kwargs):
        """
        https://pytorch.org/vision/main/generated/torchvision.transforms.functional.rotate.html
        in counter-clockwise direction
        """
        # TODO: Implement forward_mapping
        return TF.rotate(im, -90, interpolation=InterpolationMode.NEAREST)

    def inverse_view(self, noise, background=None, **kwargs):
        # TODO: Implement inverse_mapping
        return TF.rotate(noise, 90, interpolation=InterpolationMode.NEAREST)