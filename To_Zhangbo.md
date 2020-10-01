# Face_SPADE


## Install requirements firstly
```
pip install -r requirements.txt
```
If running still need other requirements, please contact me directly.


## Install SyncBN

```
cd models/networks/
git clone https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
cp -rf Synchronized-BatchNorm-PyTorch/sync_batchnorm .
cd ../../
```

## Prepare data

There are two kinds of data, which are saved in 

```
/home/jingliao/ziyuwan/workspace/dataset/CelebAMask-HQ
/home/jingliao/ziyuwan/workspace/dataset/CelebAMask-HQ-Aligned
```

Move them to anywhere you like.


## Training

Setting 2：

```
python train_face.py --name Setting_2 --load_size 256 --gpu_ids [Depends on you] --label_nc 18 --dataroot [root path of CelebAMask-HQ] --no_instance --max_dataset_size 29000 --nThreads [Depends on you] --save_epoch_freq 1 --niter 50 --niter_decay 50 --preprocess_mode resize --batchSize [Depends on you] --tensorboard_log --use_degradation_v3
```

Setting 3:

```
python train_face.py --name Setting_3 --load_size 256 --gpu_ids [Depends on you] --label_nc 18 --dataroot [root path of CelebAMask-HQ-Aligned] --no_instance --max_dataset_size 29000 --nThreads [Depends on you] --save_epoch_freq 1 --niter 50 --niter_decay 50 --preprocess_mode resize --batchSize [Depends on you] --tensorboard_log --use_degradation_v2
```
