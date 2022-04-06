import numpy as np
import os
import typing as tp

from numpy import typing as npt
from skimage import feature

from .modules import detector, classifier as clf, placer


__all__ = [
    "check_image",
    "load_polygon"
    ]


_IP_LIB_PATH = os.path.abspath(__file__)
_DEFAULT_CLR_PATH = os.path.join(_IP_LIB_PATH, "./../data/cluster_model.mod")
_DEFAULT_CLF_PATH = os.path.join(_IP_LIB_PATH, "./../data/clf_model.mod")
_n_feature_classes = 16


_defoult_fe = clf.BagOfWordExtractor("kpoint", bins=np.arange(_n_feature_classes + 1), kp_model=feature.ORB(), cluster_model=_DEFAULT_CLR_PATH)
_defoult_clf = clf.ItemClassifier(_defoult_fe, _DEFAULT_CLF_PATH)


def check_image(image: npt.ArrayLike, polygon: npt.ArrayLike,
                detect_size: tp.Tuple[int, int] = (256, 256), detect_max_dim: int = 800,        # detector params
                clf_model: tp.Any = None,                                                       # classifier params                
                place_scale: float = 4, place_step: float = 1, place_n_rots: int = 4) -> bool:  # placer params
    """checks if the objects from the image will fit in the polygon

    Args:
        image (npt.ArrayLike): input image
        polygon (npt.ArrayLike): main polygon
        detect_size (tp.Tuple[int, int], optional): detector param. Defaults to (256, 256).
        detect_max_dim (int, optional): detector param. Defaults to 800.
        place_step (float, optional): placer param. Defaults to 1.
        place_n_rots (int, optional): placer param. Defaults to 4.

    Returns:
        bool:
    """
    if clf_model is None:
        clf_model = _defoult_clf
    
    items_labels = clf_model.predict(detector.image_object_range(image, detect_size, detect_max_dim)).reshape(-1)

    return placer.try_placing(items_labels, polygon, place_scale, int(place_scale * place_step), place_n_rots)

def load_polygon(file_path: str):

    with open(file_path, "r") as mfile:
        vert_list = [list(float(vert) for vert in line[1:-2].split(", ")) for line in mfile]

    return np.array(vert_list)

