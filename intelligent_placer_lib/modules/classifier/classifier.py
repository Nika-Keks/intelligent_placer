import typing as tp
import numpy as np
import pickle

from numpy import typing as npt
from sklearn import utils as skutils

from . import image_features as imfeat


__all__ = [
    "ItemClassifier"
]


class ItemClassifier():

    def __init__(self, feature_extracrtor: imfeat.FeatureExtractor, classifier: str or tp.Any) -> None:
        """item classifier 

        Args:
            feature_extracrtor (imfeat.FeatureExtractor): feature extractor 
            classifier (str or tp.Any): classifier obaject or path to pickle dump, must have fit & predict methods
        """
        self.featire_extractor = feature_extracrtor

        if isinstance(classifier, str):
            with open(classifier, "rb") as mfile:
                self.classifier = pickle.load(mfile)
        else:
            self.classifier = classifier

    def fit(self, X: tp.Iterable[npt.ArrayLike], y: tp.Iterable):
        
        x_data = self.featire_extractor.transform(X)
        y_data = np.array(list(y))

        x_data, y_data = skutils.shuffle(x_data, y_data)

        self.classifier.fit(x_data, y_data)

    def predict(self, X: tp.Iterable[npt.ArrayLike]):
        return self.classifier.predict(self.featire_extractor.transform(X))