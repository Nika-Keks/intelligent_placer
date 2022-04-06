import typing as tp
import numpy as np
import pickle

from numpy import typing as npt
from skimage import color


__all__ = [
    "FeatureExtractor",
    "BagOfWordExtractor",
    "ColorExtractor",
    "MultiExtractor",
    "CustomClusterization"
]


class FeatureExtractor:
    
    def transform(self, X: tp.Iterable[npt.ArrayLike]) -> npt.ArrayLike:
        x_out = np.array([self._imtransform(x_item) for x_item in X])

        return x_out    
    
    def _imtransform(self, img: npt.ArrayLike) -> npt.ArrayLike:
        pass


class BagOfWordExtractor(FeatureExtractor):

    def __init__(self, mode: str, bins: npt.ArrayLike, cluster_model: str or tp.Any, kp_model: tp.Any = None, mesh_size: tp.Tuple[int, int] = None) -> None:
        super().__init__()

        if not mode in ["mesh", "kpoint"]:
            raise ValueError(f"invalid mode value: {mode}")

        self.image_features = None
        if mode == "mesh":
            self.image_features = self._mesh_features(mesh_size)
        elif mode in ["kpoint"]:
            self.image_features = self._kp_features(kp_model)          

        self.bins = bins

        if isinstance(cluster_model, str):
            with open(cluster_model, "rb") as model_file:
                self.cluster_model = pickle.load(model_file)
        elif hasattr(cluster_model, "predict"):
            self.cluster_model = cluster_model
        else:
            self.cluster_model = None

    def _imtransform(self, img: npt.ArrayLike) -> npt.ArrayLike:
        img_features = self.image_features(color.rgb2gray(img))
        if len(img_features) == 0:
            return np.zeros((len(self.bins) - 1))
        clusters_vector = self.cluster_model.predict(img_features)
        img_hist = np.histogram(clusters_vector, self.bins)[0]

        return img_hist

    @staticmethod
    def _mesh_features(mesh_size: tp.Tuple[int, int]):
        def _features(img: npt.ArrayLike):
            x_step, y_step = (img.shape[i] // mesh_size[i] for i in range(2))
            return np.array([[img[x-x_step:x, y-y_step:y].reshape(-1) for x in range(x_step, img.shape[0], x_step)] for y in range(y_step, img.shape[1], y_step)])
        
        return _features

    @staticmethod
    def _kp_features(kp_model: tp.Any):
        def _fetures(img: npt.ArrayLike):
            try:
                kp_model.detect_and_extract(img)
                return kp_model.descriptors
            except RuntimeError:
                return []
        return _fetures

    def fit(self, img_range: tp.Iterable[npt.ArrayLike]):
        x_data = []
        for img in img_range:
            x_data += list(self.image_features(color.rgb2gray(img)))
        x_data = np.array(x_data)

        self.cluster_model.fit(X=x_data)


class ColorExtractor(FeatureExtractor):

    def __init__(self, bins: npt.ArrayLike) -> None:
        super().__init__()

        self.bins = bins

    def _imtransform(self, img: npt.ArrayLike) -> npt.ArrayLike:
        color_fmaps = [img[..., i] for i in range(img.shape[-1])]
        color_hists = np.asarray([np.histogram(fmap.reshape(-1), self.bins)[0] for fmap in color_fmaps])

        return color_hists.reshape(-1)


class MultiExtractor(FeatureExtractor):

    def __init__(self, extractors: tp.List[FeatureExtractor]) -> None:
        super().__init__()

        self.extractors = extractors

    def _imtransform(self, img: npt.ArrayLike) -> npt.ArrayLike:
        features = []
        for extractor in self.extractors:
            features += list(extractor._imtransform(img))
        
        return np.array(features)


class CustomClusterization:

    def __init__(self, cluster_model: tp.Any, clf_model: tp.Any) -> None:
        self.cluster_model = cluster_model
        self.clf_model = clf_model

    def fit(self, X: npt.ArrayLike):
        y_pred = self.cluster_model.fit_predict(X)
        self.clf_model.fit(X, y_pred)

    def predict(self, X: npt.ArrayLike):
        return self.clf_model.predcit(X)