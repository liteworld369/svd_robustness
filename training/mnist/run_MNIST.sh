for denses in 0 1 2 3 4; do
	for dense_size in 128 256; do 
		for comps in 16 23 34 53 103 281 458; do
			echo "python train.py --save_dir MNIST --dataset MNIST --epoch 100 --comps $comps --method svd  --freeze 0 --normalize1 0 --normalize2 0 --denses $denses --dense_size $dense_size"
			echo "python train.py --save_dir MNIST --dataset MNIST --epoch 100 --comps $comps --method svd  --freeze 1 --normalize1 0 --normalize2 0 --denses $denses --dense_size $dense_size"
			echo "python train.py --save_dir MNIST --dataset MNIST --epoch 100 --comps $comps --method gaussian  --freeze 1 --normalize1 0 --normalize2 0 --denses $denses --dense_size $dense_size"
		done  
	done
done 