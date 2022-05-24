# code to build the features from the dataset
import pandas as pd
from skimage.io import imread
import numpy as np
from skimage.color import rgb2gray
import cv2


class build_features:
    def __init__(self, train_files, mask_files) -> None:
        self.train_files = train_files
        self.mask_files = mask_files

        self.df = pd.DataFrame({"image": self.train_files,
                                "mask": self.mask_files,
                                "label": [self.label(x) for x in mask_files]})

    def build(self):
        pass

    def features(self):
        pass

    def label(self, mask):
        value = np.max(imread(mask))
        return '1' if value > 0 else '0'

    def transform_features(self):
        image_dataset = []

        for path in self.df["image"]:
            image_dataset.append(rgb2gray(cv2.imread(path)).reshape(-1))

            image_dataset_np = np.array(image_dataset)

            image_df = pd.DataFrame(image_dataset_np)
