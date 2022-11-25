# FROM walkerlab/pytorch:python3.8-torch1.11.0-cuda11.2.1
FROM walkerlab/pytorch-jupyter:cuda-11.3.1-pytorch-1.12.0-torchvision-0.12.0-torchaudio-0.11.0-ubuntu-20.04

RUN pip3 install -U nilearn

COPY . /src/528_project
RUN pip3 install -e /src/528_project
