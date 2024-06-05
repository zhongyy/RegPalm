import torch
import numbers
import torchvision.transforms.functional as F
import warnings
from typing import List, Optional, Tuple
from collections.abc import Sequence
from torch import Tensor
import math
import PIL
from PIL import Image

def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size

class RandomResizedPaddingCrop(torch.nn.Module):
    """Crop a random portion of image and resize it to a given size.

    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions

    A crop of the original image is made: the crop has a random area (H * W)
    and a random aspect ratio. This crop is finally resized to the given
    size. This is popularly used to train the Inception networks.

    Args:
        size (int or sequence): expected output size of the crop, for each edge. If size is an
            int instead of sequence like (h, w), a square output size ``(size, size)`` is
            made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).

            .. note::
                In torchscript mode size as single int is not supported, use a sequence of length 1: ``[size, ]``.
        scale (tuple of float): Specifies the lower and upper bounds for the random area of the crop,
            before resizing. The scale is defined with respect to the area of the original image.
        ratio (tuple of float): lower and upper bounds for the random aspect ratio of the crop, before
            resizing.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.BILINEAR``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` and
            ``InterpolationMode.BICUBIC`` are supported.
            For backward compatibility integer values (e.g. ``PIL.Image[.Resampling].NEAREST``) are still accepted,
            but deprecated since 0.13 and will be removed in 0.15. Please use InterpolationMode enum.
        antialias (bool, optional): antialias flag. If ``img`` is PIL Image, the flag is ignored and anti-alias
            is always used. If ``img`` is Tensor, the flag is False by default and can be set to True for
            ``InterpolationMode.BILINEAR`` and ``InterpolationMode.BICUBIC`` modes.
            This can help making the output for PIL images and tensors closer.
    """

    def __init__(
        self,
        size,
        overcrop_input_h, 
        overcrop_input_w,
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        pad_ratio_before_crop=0.2,
        interpolation=Image.BILINEAR,
        #interpolation=InterpolationMode.BILINEAR,
    ):
        super().__init__()
        #_log_api_usage_once(self)

        assert type(size) is int
        self.size = size
        #self.size = _setup_size(size, error_msg="Please provide only two dimensions (h, w) for size.")

        if not isinstance(scale, Sequence):
            raise TypeError("Scale should be a sequence")
        if not isinstance(ratio, Sequence):
            raise TypeError("Ratio should be a sequence")
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("Scale and ratio should be of kind (min, max)")

        """
        # Backward compatibility with integer value
        if isinstance(interpolation, int):
            warnings.warn(
                "Argument 'interpolation' of type int is deprecated since 0.13 and will be removed in 0.15. "
                "Please use InterpolationMode enum."
            )
            interpolation = _interpolation_modes_from_int(interpolation)
        """
      
        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio
        self.pad_ratio_before_crop = pad_ratio_before_crop

        assert overcrop_input_h >= size and overcrop_input_w >= size
        self.overcrop_input_h = overcrop_input_h
        self.overcrop_input_w = overcrop_input_w
        self.overcrop_scale_h = overcrop_input_h / size
        self.overcrop_scale_w = overcrop_input_w / size

        assert self.pad_ratio_before_crop >= self.overcrop_scale_h - 1, "{} vs {}".format(self.pad_ratio_before_crop, self.overcrop_scale_h - 1)
        assert self.pad_ratio_before_crop >= self.overcrop_scale_w - 1, "{} vs {}".format(self.pad_ratio_before_crop, self.overcrop_scale_w - 1)

    @staticmethod
    def get_params(img: Tensor, scale: List[float], ratio: List[float]) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image or Tensor): Input image.
            scale (list): range of scale of the origin size cropped
            ratio (list): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
            sized crop.
        """
        #assert type(img) is PIL.Image
        #_, height, width = F.get_dimensions(img)
        width, height = img.size
        #assert (height, width) == (260, 270), (height, width)
        area = height * width

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
            
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.

        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        over_h, over_w = int(round(self.overcrop_scale_h * h)), int(round(self.overcrop_scale_w * w))
        assert over_h >= h and over_w >= w

        pad_size_before_crop = math.ceil(self.pad_ratio_before_crop * (h if h > w else w))
        img = F.pad(img, pad_size_before_crop)

        i += (pad_size_before_crop - (over_h - h) // 2) 
        j += (pad_size_before_crop - (over_w - w) // 2) 
        assert i >= 0 and j >= 0
        h, w = over_h, over_w

        return F.resized_crop(img, i, j, h, w, (self.overcrop_input_h, self.overcrop_input_w), self.interpolation)

    def __repr__(self) -> str:
        interpolate_str = str(self.interpolation)#self.interpolation.value
        format_string = self.__class__.__name__ + f"(size={self.size}"
        format_string += f", overcrop_input_h={self.overcrop_input_h}"
        format_string += f", overcrop_input_w={self.overcrop_input_w}"
        format_string += f", scale={tuple(round(s, 4) for s in self.scale)}"
        format_string += f", ratio={tuple(round(r, 4) for r in self.ratio)}"
        format_string += f", interpolation={interpolate_str})"
        #format_string += f", antialias={self.antialias})"
        return format_string
