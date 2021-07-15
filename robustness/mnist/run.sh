INPUT=combined_training.csv
OLDIFS=$IFS
IFS=','
[ ! -f $INPUT ] && { echo "$INPUT file not found"; exit 99; }
while IFS=, read -r filename model_index val_sparse_categorical_accuracy
do
	eval $(echo "File:$filename")
	for trials in 1 10; do
		for steps in 40 100; do
			for eps in 1.0 2.0; do
				eval $(echo "python main.py --gpu 0 --fname $filename --trials $trials --steps $steps --eps $eps --norm l2") | tee -a output.csv
			done
			for eps in 0.3 0.15; do
				eval $(echo "python main.py --gpu 0 --fname $filename --trials $trials --steps $steps --eps $eps --norm linf") | tee -a output.csv
			done
		done
	done
done < combined_training.csv 
IFS=$OLDIFS

read line