import numpy as np
import typing as tp

from skimage import draw, transform
from numpy import typing as npt

from dataclasses import dataclass

__all__ = [
    "try_placing"
]

@dataclass
class ItemData:
    name: str
    circum_polygon: npt.ArrayLike

items_data = {
    1: ItemData("powerbank",    np.array([ # 10.5 x 2
        [0, 0], [10.5, 0], [10.5, 2], [0, 2]
    ])),
    2: ItemData("trinket",      np.array([ # 5 x 2
        [0, 0], [5, 0], [5, 2], [2, 0]
    ])),
    3: ItemData("phone",        np.array([ # 13.2 x 7
        [0, 0], [13.2, 0], [13.2, 7], [0, 7]
    ])),
    4: ItemData("pen",          np.array([ # 14 x 0.7
        [0 ,0], [14, 0], [14, 0.7], [0, 0.7]
    ])),
    5: ItemData("card",         np.array([ # 8.5 x 5.3
        [0 ,0], [8.5, 0], [8.5, 5.3], [0, 5.3]
    ])),
    6: ItemData("record_book",  np.array([ # 14.3 x 10.5
        [0 ,0], [14.3, 0], [14.3, 10.5], [0, 10.5]
    ])),
    7: ItemData("mediator",     np.array([ # 
        [1.5, 0], [0, 1], [1.5, 3], [3, 1]
    ])),
    8: ItemData("coin",         np.array([ # 2 x 2
        [0, 0], [2, 0], [2, 2], [0, 2]
    ])),
    9: ItemData("glasses_case", np.array([ # 14.9 x 5.5
        [0, 0], [14.9, 0], [14.9, 5.5], [0, 5.5]
    ])),
    10:ItemData("lighter",      np.array([ # 5.6 x 3.6
        [0, 0], [5.6, 0], [5.6, 3.6], [0, 3.6]
    ]))
}

def try_placing(items_labels: npt.ArrayLike, polygon: npt.ArrayLike, scale: float,
                shift_step: int, n_rots: int) -> bool:
    """checks if the objects from the image will fit in the polygon

    Args:
        items_labels (npt.ArrayLike): classes labels array
        polygon (npt.ArrayLike): input polygon
        scale (float): scale faktor for masking
        shift_step (int): shift grid param
        n_rots (int): rotate grid param

    Returns:
        bool:
    """
    loop_stack = [ 
        (_polygon2image(polygon, scale), (0, 0), 0, 0) 
        ]

    while len(loop_stack) != 0:
        dpol, shift, rot, itemi = loop_stack.pop()

        if itemi >= len(items_labels):
            return True

        item_dpol = _polygon2image(items_data[items_labels[itemi]].circum_polygon, scale)
        success,  next_dpol, shift, rot = _single_placing(dpol, item_dpol, 
                                                                    shift, shift_step,
                                                                    rot, n_rots)
        if success:
            loop_stack.append((dpol, shift, rot, itemi))
            loop_stack.append((next_dpol, (0, 0), 0, itemi+1))
    
    return False

def _polygon2image(polygon: npt.ArrayLike, scale: float) -> npt.ArrayLike:

    std_polygon = ((polygon - np.min(polygon, axis=0)) * scale).astype(np.uint16)
    image = np.zeros(np.max(std_polygon, axis=0) + 1, dtype=np.bool8)
    rows, cols = draw.polygon(*[std_polygon[:, i] for i in range(2)])
    image[rows, cols] = True

    return image

def _single_placing(inclusive_dpol: npt.ArrayLike, item_dpol: npt.ArrayLike, 
                    start_shift: tp.Tuple[int, int], shift_step: int,
                    start_rot: int, n_rots: int) -> tp.Tuple[bool, npt.ArrayLike, int, int]:

    if np.sum(inclusive_dpol) < np.sum(item_dpol):
        return False, None, None, None

    for current_rot, angle in zip(range(start_rot, n_rots), np.linspace(0, 180, n_rots, endpoint=False)[start_rot:]):
        rot_item_dpol = transform.rotate(item_dpol, angle, resize=True)

        if not all(map(lambda x, y: x >= y, inclusive_dpol.shape, rot_item_dpol.shape)):
            continue
        
        stop_shift = list(map(lambda x, y: x - y, inclusive_dpol.shape, rot_item_dpol.shape))

        for i in range(start_shift[0], stop_shift[0], shift_step):
            for j in range(start_shift[1], stop_shift[1], shift_step):

                success = np.all(np.logical_or(inclusive_dpol[i: i+rot_item_dpol.shape[0], j:j+rot_item_dpol.shape[1]], np.logical_not(rot_item_dpol)))

                if success:
                    in_dpol_copy = np.copy(inclusive_dpol)
                    in_dpol_copy[i:i+rot_item_dpol.shape[0], j:j+rot_item_dpol.shape[1]][rot_item_dpol] = False
                    
                    return True, in_dpol_copy, (i, j), current_rot
    
    return False, None, None, None