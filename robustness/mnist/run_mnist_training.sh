INPUT=MNIST_training.csv
OLDIFS=$IFS
IFS=','
[ ! -f $INPUT ] && { echo "$INPUT file not found"; exit 99; }
while IFS=, read -r filename model_index val_sparse_categorical_accuracy
do
	for trials in 1 10; do
		for steps in 40 100; do
			for eps in 10.0 50.0 100.0; do
				echo "python main.py --dataset MNIST --attack pgd --fname $filename --trials $trials --steps $steps --eps $eps --norm l1"
			done
			for eps in 1.0 1.5 2.0; do
				echo "python main.py --dataset MNIST --attack pgd --fname $filename --trials $trials --steps $steps --eps $eps --norm l2"
			done
			for eps in 0.1 0.15 0.2; do
				echo "python main.py --dataset MNIST --attack pgd --fname $filename --trials $trials --steps $steps --eps $eps --norm linf"
			done
		done
	done
done < MNIST_training.csv 
IFS=$OLDIFS