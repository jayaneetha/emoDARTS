import socket

import torch

DEVICE = 'cpu'

if torch.cuda.is_available():
    DEVICE = 'cuda'

host = socket.gethostname()

IEMOCAP_DIR = ''
ESD_DIR = ''
MSPIMPROV_DIR = ''
MSPPODCAST_DIR = ''
