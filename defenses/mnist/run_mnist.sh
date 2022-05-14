for method in svd; do
	for comps in 16 23 34 53 103 281 458; do  
		for denses in 0 1 2 4; do
			for dense_size in 256; do 
				eval $(echo "python train.py --save_dir MNIST --dataset MNIST --epoch 100 --comps $comps --method $method --freeze 1 --reconstruct 2 --normalize1 0 --normalize2 0 --denses $denses --dense_size $dense_size") 
			done  
		done
	done   
done