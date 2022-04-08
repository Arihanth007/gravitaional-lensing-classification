import os
import numpy as np
from PIL import Image
from typing import List, Tuple
from bisect import bisect_right
from torch.utils.data import Dataset


class CustomImageDataset(Dataset):
    def __init__(self, path: str, classes: List[str]) -> None:
        self.classes = classes
        self.paths = {c: f"{path}/{c}" for c in self.classes}
        self.count = {c: len(next(os.walk(self.paths[c]))[
                             2]) for c in self.classes}

    def __len__(self) -> int:
        return sum(self.count.values())

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        cum_sum = np.cumsum([self.count[c] for c in self.classes])
        class_idx = bisect_right(cum_sum, idx)

        class_name = self.classes[class_idx]
        img_idx = idx + 1 - (cum_sum[class_idx-1] if class_idx else 0)

        img_path = f"{self.paths[class_name]}/{img_idx}.npy"

        X = np.load(img_path)
        X = np.concatenate((X, X, X), axis=0)*255
        y = class_idx
        return X, y


class LensingData(Dataset):

    def __init__(self, data, transform=None):
        self.transform = transform
        self.images = data[:, 0]
        self.labels = data[:, 1]
        self.num_samples = len(self.labels)

    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.labels)
