import numpy as np
import typing as tp

from numpy import typing as npt
from scipy import ndimage
from skimage import feature, color, morphology, transform, measure


__all__ = [
    "image_object_range",
    ]


def image_object_range(img: npt.ArrayLike, size: tp.Tuple[int, int], max_dim: int):
    """range of objects on input image

    Args:
        img (npt.ArrayLike): input image
        size (tp.Tuple[int, int]): output images size
        max_dim (int): max dimention of image for processing

    Yields:
        _type_: image object
    """
    if img.dtype == np.uint8:
        img = img / 255

    mask = _oblects_mask(img, max_dim=max_dim)
    imlabel = measure.label(mask)

    for i in range(1, max(imlabel.reshape(-1)) + 1):
        yield transform.resize(_extract_object(img, imlabel==i), size)

def _insert_in_squar(img: npt.ArrayLike, bg_color: float):

    new_size = int(max(img.shape) * np.sqrt(2))
    new_img = np.zeros((new_size, new_size, 3))
    new_img[...] = bg_color
    
    find_shift = lambda i: (new_img.shape[i] - img.shape[i]) // 2
    x_shift, y_shift = (find_shift(i) for i in [0, 1])
    new_img[x_shift:x_shift+img.shape[0], y_shift:y_shift+img.shape[1]] = img

    return new_img

def _extract_object(img: npt.ArrayLike, mask: npt.ArrayLike):

    vertical_indices = np.where(np.any(mask, axis=1))[0]
    top, bottom = max([0, vertical_indices[0]]), min([mask.shape[0], vertical_indices[-1]])

    horizontal_indices = np.where(np.any(mask, axis=0))[0]
    left, right = max([0, horizontal_indices[0]]), min([mask.shape[1], horizontal_indices[-1]])    

    masked_img = img * color.gray2rgb(mask)
    bg_color = np.mean(img.reshape(-1)[color.gray2rgb(np.logical_not(mask)).reshape(-1)])
    masked_img[color.gray2rgb(np.logical_not(mask))] = bg_color

    croped_img = masked_img[top:bottom, left:right]
    
    squared_img = _insert_in_squar(croped_img, bg_color)
    
    return squared_img

def _oblects_mask(img: npt.ArrayLike, max_dim: int):
   
    imgray = transform.rescale(color.rgb2gray(img), scale=max_dim / max(img.shape))
    
    imborder = morphology.binary_closing(feature.canny(imgray, sigma=1.5, mode="nearest"), footprint=np.ones((4, 4)))
    imsegm = ndimage.morphology.binary_fill_holes(imborder)
    imsegm = morphology.binary_opening(imsegm, footprint=np.ones((4, 4)))
    
    return  transform.resize(imsegm, output_shape=img.shape[:2])