for comps in 10 20 30 40 50 60 70 80 90 100; do 
	eval $(echo "python train.py --gpu 0 --original true --comps $comps")
done
read line