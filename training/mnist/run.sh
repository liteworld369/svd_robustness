for freeze in 0 1; do
	for normalize in 1 0; do 
			for comps in 10 20 30 40 50 60 70 80 90 100; do
				eval $(echo "python train.py --gpu 0 --comps $comps --normalize $normalize --freeze $freeze")
			done
		done
	done
done
read line