# Dockerfile with training functionality

FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

# set the working directory in the container
RUN mkdir app
WORKDIR app/

RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl \
&& apt-get install unzip \
&& apt-get -y install python3 \
&& apt-get -y install python3-pip \
&& apt-get install ffmpeg libsm6 libxext6  -y \
&& apt-get install wget \
&& apt-get clean \
&& pip3 -q install pip --upgrade 


ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV GIT_PYTHON_REFRESH=quiet

RUN apt-get update && apt-get install -y \
  curl \
  && rm -rf /var/lib/apt/lists/*


# install miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py37_4.9.2-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc
ENV PATH /opt/conda/bin:$PATH

# copy the dependencies file to the working directory - repair after testing !!!
COPY ./Dockerfiles/train_models/requirements.txt .
COPY ./setup.py .

RUN pip3 install argparse && \
    python3 setup.py develop 
    
COPY ./RedMask ./RedMask

ENV PYTHONPATH /app/


