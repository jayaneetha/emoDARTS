import random
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from constant import MSPPODCAST_DIR


class MspPodcastDatasetKFold(Dataset):
    def __init__(self, **kwargs) -> None:
        self.data = None
        f = kwargs['features']
        fold = kwargs['fold']
        if kwargs['train']:
            path = f"{MSPPODCAST_DIR}/processed_{f}_k_fold/{fold}/{f}/train"
        else:
            path = f"{MSPPODCAST_DIR}/processed_{f}_k_fold/{fold}/{f}/val"
        self.path = Path(path)
        self.class_names = sorted(set(p.name for p in self.path.iterdir() if p.is_dir()))
        self.class_ids_for_name = dict((name, idx) for idx, name in enumerate(self.class_names))

        spectrums, self.classes = self.get_files_and_class_names()
        self.pairs = list(zip(spectrums, self.classes))
        random.shuffle(self.pairs)
        # Take only 10000 utterances
        if len(self.pairs) > 10000:
            self.pairs = self.pairs[0:10000]

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

    def get_class_weights(self):

        frequency = {}

        # iterating over the list
        for item in self.classes:
            # checking the element in dictionary
            if item in frequency:
                # incrementing the count
                frequency[item] += 1
            else:
                # initializing the count
                frequency[item] = 1

        # Compute the frequency of each label
        label_frequencies = {label: count / self.__len__() for label, count in frequency.items()}

        # Calculate the inverse weight for each label
        key_weights = {label: 1.0 / freq for label, freq in label_frequencies.items()}

        # calculate the inverse count for each label
        count_weights = {label: 1.0 / count for label, count in frequency.items()}

        # Create a list of dictionaries containing the 'label', 'frequency', and 'weight' for each label
        weights_list = []
        for label in frequency.keys():
            frequency = label_frequencies[label]
            weight = key_weights[label]
            c_weight = count_weights[label]
            weights_list.append({'label': label, 'frequency': frequency, 'weight': weight, 'c_weight': c_weight})

        # Print the list of dictionaries
        # print(weights_list)

        return_weights = {}
        for l in self.class_ids_for_name.keys():
            return_weights[self.class_ids_for_name[l]] = key_weights[l]

        return return_weights
