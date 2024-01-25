import numpy as np


def downsample_with_max_pooling(array, factor=(1, 4)):
    if np.all(np.array(factor, int) == 1):
        return array

    sections = []

    for offset in np.ndindex(factor):
        part = array[tuple(np.s_[o::f] for o, f in zip(offset, factor))]
        sections.append(part)

    output = sections[0].copy()

    for section in sections[1:]:
        if output.shape == section.shape:
            np.maximum(output, section, output)
        else:
            if output.shape[0] != section.shape[0]:
                c = output.shape[0] - section.shape[0]
                pad = np.zeros((c, output.shape[1]))
                s = np.vstack((section, pad))
                np.maximum(output, s, output)
            if output.shape[1] != section.shape[1]:
                c = output.shape[1] - section.shape[1]
                pad = np.zeros((output.shape[0], c))
                s = np.hstack((section, pad))
                np.maximum(output, s, output)

    return output
