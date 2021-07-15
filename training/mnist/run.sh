for dense_size in 128 256 512; do
	for comps in 10 20 30 40 50 60 70 80 90 100; do
	 eval $(echo "python train.py --gpu 0 --original true --comps $comps --dense_size $dense_size")
	 for reconstruct in 'true' 'false'; do
		eval $(echo "python train.py --gpu 0 --original false --comps $comps --reconstruct $reconstruct --dense_size $dense_size")
	 done
	done
done
read line