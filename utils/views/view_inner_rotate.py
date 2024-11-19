import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms import InterpolationMode

from .view_base import BaseView

def get_circle_mask(img_size: int, r: int):
    # TODO: Implement get_circle_mask
    # raise NotImplementedError(f"get_circle_mask not implemented")
    center = img_size // 2
    
    y, x = np.ogrid[:img_size, :img_size]
    distance_from_center = np.sqrt((x - center) ** 2 + (y - center) ** 2)
    mask = distance_from_center <= r

    mask = torch.tensor(mask, dtype=torch.float32)
    return mask.unsqueeze(0)


def inner_rotate_func_with_mask(im: torch.Tensor, mask: torch.Tensor, angle, interpolate=False):
    # TODO: Implement inner_rotate_func_with_mask
    
    original_img = im.clone()
    rot_img = TF.rotate(im, -angle, interpolation=InterpolationMode.BILINEAR)  # clockwise-manner
    interpolated_img = rot_img * mask + original_img * (1 - mask)
    return interpolated_img
    
 
class InnerRotateView(BaseView):
    """
    Implements an "inner circle" view, where a circle inside the image spins
    but the border stays still. Inherits from `PermuteView`, which implements
    the `view` and `inverse_view` functions as permutations. We just make
    the correct permutation here, and implement the `make_frame` method
    for animation
    """

    def __init__(self, angle):
        """
        Make the correct "inner circle" permutations and pass it to the
        parent class constructor.
        """
        self.angle = angle
        self.stage_1_mask = get_circle_mask(64, 24)
        self.stage_2_mask = get_circle_mask(256, 96)

    def view(self, im, **kwargs):
        im_size = im.shape[-1]
        if im_size == 64:
            mask = self.stage_1_mask.to(im)
            self.stage_1_mask = mask
        elif im_size == 256:
            mask = self.stage_2_mask.to(im)
            self.stage_2_mask = mask

        inner_rotated = inner_rotate_func_with_mask(im, mask, -self.angle, interpolate=False)

        return inner_rotated

    def inverse_view(self, im, **kwargs):
        im_size = im.shape[-1]
        if im_size == 64:
            mask = self.stage_1_mask.to(im)
            self.stage_1_mask = mask
        elif im_size == 256:
            mask = self.stage_2_mask.to(im)
            self.stage_2_mask = mask

        inner_rotated = inner_rotate_func_with_mask(im, mask, self.angle, interpolate=False)

        return inner_rotated

