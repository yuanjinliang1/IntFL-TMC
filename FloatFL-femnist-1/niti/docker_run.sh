NITI_PATH="/home/yuan/code/niti-float/niti"
DATA_PATH="/data/shared_datasets/federated_dataset"
docker build -t wangmaolin/niti:0.1 .
docker run --gpus all -v $NITI_PATH:/niti -v $DATA_PATH:/fed_data -it wangmaolin/niti:0.1

