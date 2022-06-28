python train.py --save_dir FMNIST --dataset FMNIST --epoch 50 --comps 4 --method svd  --freeze 1 --regularizer 0 --denses 1 --dense_size 256 --gpu 1 &
python train.py --save_dir FMNIST --dataset FMNIST --epoch 50 --comps 4 --method svd  --freeze 1 --regularizer 1 --denses 1 --dense_size 256 --gpu 1 &
python train.py --save_dir FMNIST --dataset FMNIST --epoch 50 --comps 65 --method svd  --freeze 1 --regularizer 0 --denses 1 --dense_size 256 --gpu 1 &
python train.py --save_dir FMNIST --dataset FMNIST --epoch 50 --comps 65 --method svd  --freeze 1 --regularizer 1 --denses 1 --dense_size 256 --gpu 1 &
