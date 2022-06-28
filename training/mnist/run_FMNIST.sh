for denses in 1; do
	for dense_size in 256; do 
		for comps in 4 65; do
			echo "python train.py --save_dir FMNIST --dataset FMNIST --epoch 50 --comps $comps --method svd  --freeze 1 --regularizer 0 --denses $denses --dense_size $dense_size"
			echo "python train.py --save_dir FMNIST --dataset FMNIST --epoch 50 --comps $comps --method svd  --freeze 1 --regularizer 1 --denses $denses --dense_size $dense_size"
		done  
	done
done 