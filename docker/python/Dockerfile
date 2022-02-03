FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update -y && \
    apt-get install -y rsync

ARG UID
RUN useradd -l -m -u ${UID} dockeruser
USER ${UID}
ENV PATH=$PATH:/home/dockeruser/.local/bin

RUN pip install --user joblib tqdm scikit-learn matplotlib
