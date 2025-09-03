NITI_PATH="/data2/yjl/code/fedupdate/femnist/MixPrecisionFL-femnist-1/niti"
DATA_PATH="/data6T/YuanJinliang/data"
docker build -t yuanjinliang-fedupdate-femnist/niti:0.1 .
docker run --gpus all --name yjl_fedupdate_femnist -v $NITI_PATH:/niti -v $DATA_PATH:/fed_data -it yuanjinliang-fedupdate-femnist/niti:0.1

