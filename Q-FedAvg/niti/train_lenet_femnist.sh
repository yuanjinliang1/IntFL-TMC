# DEPTH=13
    CUDA_VISIBLE_DEVICES=0\
    python ti_main.py 
    # --model-type int\
	# --dataset cifar10\
	# --model lenet\
    # --depth $DEPTH\
	# --data-dir /niti\
	--results-dir ./results --save test\
	# --epochs 150\
	# --epochs 150\
	# --batch-size 5\
	# -j 8\
	# --log-interval 20\
	# --weight-decay\
 	# --init /niti/cifar10_vgg"$DEPTH"_rebalance_init.pth.tar\
    # --download\
