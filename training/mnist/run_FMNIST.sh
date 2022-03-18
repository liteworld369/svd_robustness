for denses in 0 1 2 3 4; do
	for dense_size in 128 256; do 
		for comps in 2 4 6 16 65 319 619; do
			echo "python train.py --save_dir FMNIST --dataset FMNIST --epoch 100 --comps $comps --method svd  --freeze 0 --normalize1 0 --normalize2 0 --denses $denses --dense_size $dense_size"
			echo "python train.py --save_dir FMNIST --dataset FMNIST --epoch 100 --comps $comps --method svd  --freeze 1 --normalize1 0 --normalize2 0 --denses $denses --dense_size $dense_size"
			echo "python train.py --save_dir FMNIST --dataset FMNIST --epoch 100 --comps $comps --method gaussian  --freeze 1 --normalize1 0 --normalize2 0 --denses $denses --dense_size $dense_size"
		done  
	done
done 