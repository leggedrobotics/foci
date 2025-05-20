FROM nvidia/cuda:12.2.0-base-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# Update the package lists
RUN apt-get update


ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install dependencies for Open3D
RUN apt-get install -y build-essential
RUN apt-get install -y cmake
RUN apt-get install -y git
RUN apt-get install -y libgl1-mesa-glx
RUN apt-get install -y libgl1-mesa-dev
RUN apt-get install -y libglfw3-dev
RUN apt-get install -y libssl-dev
RUN apt-get install -y libusb-1.0-0-dev
RUN apt-get install -y libudev-dev
RUN apt-get install -y python3-dev
RUN apt-get install -y python3-pip
RUN apt-get install -y wget
RUN apt-get install -y  tmux


RUN pip3 install numpy
RUN pip3 install open3d
RUN pip3 install matplotlib


RUN apt install -y gcc g++ gfortran git cmake liblapack-dev pkg-config --install-recommends
RUN apt install -y ipython3 python3-dev python3-numpy python3-scipy python3-matplotlib --install-recommends
RUN apt install -y swig --install-recommends
RUN apt install -y libblas-dev liblapack-dev libmetis-dev


# Install HSL
WORKDIR /
RUN git clone https://github.com/coin-or-tools/ThirdParty-HSL.git
COPY coinhsl /ThirdParty-HSL/coinhsl
WORKDIR /ThirdParty-HSL
RUN ./configure
RUN make
RUN make install

ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
RUN apt-get install -y rename
RUN rename 's/libhsl/libcoinhsl/g' /usr/local/lib*


# Install IPOPT
WORKDIR /
RUN apt-get install -y git cmake gcc g++ gfortran pkg-config
RUN git clone https://github.com/coin-or/Ipopt.git 
WORKDIR /Ipopt
RUN ./configure
RUN make -j8
RUN make test
RUN make install


WORKDIR /
RUN git clone https://github.com/casadi/casadi.git
RUN apt-get install -y software-properties-common
RUN apt-get update

WORKDIR casadi
RUN cmake -DWITH_PYTHON=ON -DWITH_PYTHON3=ON -DWITH_OPENMP=ON -DWITH_IPOPT=ON -DWITH_HSL=ON  
RUN make
RUN make install

RUN apt-get install -y python-is-python3


WORKDIR /
RUN pip install warp-lang
RUN wget https://github.com/NVIDIA/warp/releases/download/v1.0.2/warp_lang-1.0.2-py3-none-manylinux2014_x86_64.whl
RUN pip install warp_lang-1.0.2-py3-none-manylinux2014_x86_64.whl

RUN pip install usd-core

RUN apt-get install -y vim

RUN pip3 install viser


# # expose 8888 for jupyter notebook
EXPOSE 8080

WORKDIR /workspace
COPY . /workspace
