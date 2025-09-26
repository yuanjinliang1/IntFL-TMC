NITI_PATH="/data2/yjl/code/FloatFL/cifar10/FloatFL-cifar10-1/niti"
docker build -t wangmaolin/niti:0.1 .
docker run --gpus all -v $NITI_PATH:/niti -it wangmaolin/niti:0.1

