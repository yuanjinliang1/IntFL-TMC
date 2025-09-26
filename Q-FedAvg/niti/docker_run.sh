NITI_PATH="/data2/yjl/code/IntFL/femnist/IntFL-femnist-1/niti"
DATA_PATH="/data/shared_datasets/federated_dataset"
docker build -t wangmaolin/niti:0.1 .
docker run --gpus all --name yjl_INT8FL -v $NITI_PATH:/niti -v $DATA_PATH:/fed_data -it wangmaolin/niti:0.1
