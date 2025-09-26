# Towards Energy-efficient Federated Learning via INT8-based Training on Mobile DSPs

## Introduction
Federated Learning (FL) enables training machine learning models across decentralized edge devices while keeping data localized. 
However, FL's significant computational demands can lead to high energy consumption and reduced battery life on mobile devices. 
This project introduces a novel framework that leverages ​8-bit integer (INT8) quantization​ for efficient neural network training directly on mobile ​Digital Signal Processors (DSPs)​. 
Our core contribution demonstrates that this approach can achieve substantial energy savings with only a minimal impact on model accuracy, paving the way for more sustainable and practical FL deployments on resource-constrained devices.

## Project Structure
project-root/
├── data/           # Scripts for dataset downloading, preprocessing, and partitioning
├── src/            # Core source code for the FL framework and quantization algorithms
├── models/         # Definitions of neural network models used in the experiments
├── experiments/    # Configuration files and scripts to reproduce the paper's experiments
├── utils/          # Utility functions (e.g., energy monitoring, logging, metrics calculation)
├── docs/           # Additional documentation
└── requirements.txt # Python dependencies


## Usage
### Step 1: Install Nvidia docker
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

### Step 2: Download NITI
> git clone https://github.com/wangmaolin/niti.git

### Step 3: Modify the path in docker_run.sh 
Change NITI_PATH in docker_run.sh to the location where you download NITI.

### Step 4: Launch the docker at $NITI_PATH
> docker_run.sh

### Step 5: Install the tensor core extension
Inside the docker, run:
> make install

## An example of training int8 VGG on CIFAR10 
> ./train_vgg_cifar10.sh


## Notes
Our current implementation utilizes cuBLAS(10.1) to run int8 matrix multiply operations directly on tensor cores.
However, there are no direct supports for int8 **batched** matrix multiply and 2D convolution even in the latest cuBLAS(11.0) and cuDNN(8.0).
So it hasn't reached the full acceleration potential of the idea that trains neural networks with integer-only arithmetic yet.
For now, it only serves as a prototype for the idea.

## Some key codes
### ti_torch.py 
Implementation of convolution, fully connected layer with int8 forward pass and backward pass 

### pytorch/tcint8mm-extension
CUDA extension using tensor core to accelerate 8 bit signed integer matrix multiply

### pytorch/int8im2col-extension
CUDA extension doing 8 bits integer image to column operation
