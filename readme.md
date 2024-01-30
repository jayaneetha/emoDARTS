# emoDARTS

## Joint Optimisation of CNN & Sequential Neural Network Architectures for Superior Speech Emotion Recognition

This repository contains the code for the
paper [emoDARTS: Joint Optimisation of CNN & Sequential Neural Network Architectures for Superior Speech Emotion Recognition](https://arxiv.org/abs/2305.14402)
by [Thejan Rajapakshe et al.]

The code is originally forked from [khanrc/pt.darts](https://github.com/khanrc/pt.darts)

### Abstract

Speech Emotion Recognition (SER) is critical in allowing emotion-aware communication in human-computer interactions.
Recent Deep Learning (DL) developments have significantly improved the performance of SER models by increasing model
complexity. However, creating an optimum DL architecture necessitates prior expertise and experimental assessments.
Neural Architecture Search (NAS), on the other hand, provides a potential path for automatically determining an ideal DL
model. Differentiable Architecture Search (DARTS) in particular is an efficient technique to discover optimal models.
This research introduces emoDARTS, a DARTS-optimised joint CNN and Sequential Neural Network (SeqNN: LSTM, RNN)
architecture that enhances SER performance, where the literature informs the selection of CNN and LSTM coupling to
deliver improved performance. While DARTS has previously been used to choose CNN and LSTM operations independently, our
technique adds a novel mechanism in selecting CNN and SeqNN operations in conjunction using DARTS. Unlike earlier work,
we do not impose limits on the layer order of the CNN. Instead, we let DARTS choose the best layer order inside the
DARTS cell on its own. We show that emoDARTS outperforms humans in designing CNN-LSTM models and surpasses the
best-reported SER results achieved through DARTS on CNN-LSTM by utilising the IEMOCAP, MSP-IMPROV, and MSP-Podcast
datasets.

### Citation

If you find this repository useful in your research, please cite:

```
@misc{rajapakshe2024enhancing,
      title={Enhancing Speech Emotion Recognition Through Differentiable Architecture Search}, 
      author={Thejan Rajapakshe and Rajib Rana and Sara Khalifa and Berrak Sisman and Björn Schuller},
      year={2024},
      eprint={2305.14402},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}
```
