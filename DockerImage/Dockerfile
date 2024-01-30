FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-devel

RUN apt update 
RUN apt install -y wget nano git

EXPOSE 8888

RUN conda install -c conda-forge jupyterlab
RUN pip install tensorboard==2.11.2
RUN pip install torchaudio==0.13.1
RUN pip install torchvision==0.14.0

RUN apt install -y libsndfile1

RUN pip install librosa==0.9.2

RUN apt -y install graphviz
RUN pip install graphviz
RUN conda install python-graphviz


CMD ["bash"]