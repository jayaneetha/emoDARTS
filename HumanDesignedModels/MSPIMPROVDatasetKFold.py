import random
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from constant import MSPIMPROV_DIR


class MspImprovDatasetKFold(Dataset):
    def __init__(self, **kwargs) -> None:
        self.data = None
        f = kwargs['features']
        fold = kwargs['fold']
        if kwargs['train']:
            path = f"{MSPIMPROV_DIR}/processed_{f}_k_fold/{fold}/{f}/train"
        else:
            path = f"{MSPIMPROV_DIR}/processed_{f}_k_fold/{fold}/{f}/val"
        self.path = Path(path)
        self.class_names = sorted(set(p.name for p in self.path.iterdir() if p.is_dir()))
        self.class_ids_for_name = dict((name, idx) for idx, name in enumerate(self.class_names))

        spectrums, classes = self.get_files_and_class_names()
        self.pairs = list(zip(spectrums, classes))
        random.shuffle(self.pairs)

    def get_files_and_class_names(self):
        spectrum_paths = list(self.path.glob('*/*.npy'))
        classes = [p.parent.name for p in spectrum_paths]

        spectrums = []
        for sp in spectrum_paths:
            d = np.load(sp)
            spectrums.append(d.reshape((1, d.shape[0], d.shape[1])).astype('float32'))

        self.data = np.array(spectrums)

        return spectrums, classes

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        return self.pairs[index][0], self.class_ids_for_name[self.pairs[index][1]]
