NITI_PATH="/data2/yjl/code/IntFL/femnist/IntFL-femnist-1/niti"
DATA_PATH="/data/shared_datasets/federated_dataset"
docker build -t wangmaolin/niti:0.1 .
docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 --shm-size 8G --gpus all -v $NITI_PATH:/niti -v $DATA_PATH:/fed_data -it wangmaolin/niti:0.1