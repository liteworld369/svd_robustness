INPUT=FMNIST_defenses.csv
OLDIFS=$IFS
IFS=','
[ ! -f $INPUT ] && { echo "$INPUT file not found"; exit 99; }
while IFS=, read -r filename model_index val_accuracy
do
	for trials in 1 10; do
		for steps in 40 100; do
			for eps in 0.0 10.0 50.0 100.0; do
				echo "python main.py --dataset FMNIST --attack pgd --fname $filename --trials $trials --steps $steps --eps $eps --norm l1"
			done
			for eps in 0.0 1.0 1.5 2.0; do
				echo "python main.py --dataset FMNIST --attack pgd --fname $filename --trials $trials --steps $steps --eps $eps --norm l2"
			done
			for eps in 0.0 0.1 0.15 0.2; do
				echo "python main.py --dataset FMNIST --attack pgd --fname $filename --trials $trials --steps $steps --eps $eps --norm linf"
			done
		done
	done
done < FMNIST_defenses.csv 
IFS=$OLDIFS