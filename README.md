# Towards Energy-efficient Federated Learning via INT8-based Training on Mobile DSPs

## Introduction
Federated Learning (FL) enables training machine learning models across decentralized edge devices while keeping data localized. 
However, FL's significant computational demands can lead to high energy consumption and reduced battery life on mobile devices. 
This project introduces a novel framework that leverages ​8-bit integer (INT8) quantization​ for efficient neural network training directly on mobile ​Digital Signal Processors (DSPs)​. 
Our core contribution demonstrates that this approach can achieve substantial energy savings with only a minimal impact on model accuracy, paving the way for more sustainable and practical FL deployments on resource-constrained devices.

## Project Structure
project-root/ <br>
├── Data/           # Scripts for dataset downloading, preprocessing, and partitioning <br>
├── On-device cost/            # Source code for the training latency and energy measurement on mobile devices <br>
├── FloatFL/         # Baseline: the traditional FP32-based FL protocol with FedAvg <br>
├── QuanFL-SignSGD/    # Baseline: reducing the network transmission time through model quantization as INT8 format, but still using FP32-based training on devices <br>
├── Q-FedAvg/          # Baseline: directly integrating INT8 training with FedAvg <br>
├── Q-FedUpdate/           # Our proposed algorithm <br>


## Usage: for each algorithm (FloatFL, QuanFL-SignSGD, Q-FedAvg, Q-FedUpdate)
### Step 1: Install Nvidia docker
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

### Step 2: Modify the path in docker_run.sh 
Change NITI_PATH in docker_run.sh to the location where you download NITI.

### Step 3: Launch the docker at $NITI_PATH
> docker_run.sh

### Step 4: Install the tensor core extension
Inside the docker, run:
> make install
