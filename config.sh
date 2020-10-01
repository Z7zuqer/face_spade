sudo apt-get update
sudo apt-get --yes install vim
yes | sudo apt-get --yes install tmux

# sudo git reset --hard HEAD
# sudo git pull https://raywzy:wanziyu888@github.com/raywzy/Face_SPADE.git
cd /mnt/blob/old_photo/Face_SPADE/
git pull https://raywzy:wanziyu888@github.com/raywzy/Face_SPADE.git
pip install -r requirements.txt

cd models/networks/
git clone https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
cp -rf Synchronized-BatchNorm-PyTorch/sync_batchnorm .
cd ../../

wandb login 8f069d8b7c58a088c057788d673beaaa8afbe3c2

# python train_face.py --name Setting_2 --load_size 256 --gpu_ids 0,1,2,3 --label_nc 18 --dataroot ../CelebAMask-HQ --no_instance --max_dataset_size 29000 --nThreads 8 --save_epoch_freq 1 --niter 50 --niter_decay 50 --preprocess_mode resize --batchSize 16 --tensorboard_log --use_degradation_v3

# python train_face.py --name Setting_3 --load_size 256 --gpu_ids 0,1,2,3 --label_nc 18 --dataroot ../CelebAMask-HQ-Aligned --no_instance --max_dataset_size 29000 --nThreads 8 --save_epoch_freq 1 --niter 50 --niter_decay 50 --preprocess_mode resize --batchSize 16 --tensorboard_log --use_degradation_v2

python train_face_wandb.py --name Setting_2_new --load_size 256 --gpu_ids 0,1,2,3 --label_nc 18 --dataroot ../CelebAMask-HQ --no_instance --max_dataset_size 29000 --nThreads 8 --save_epoch_freq 1 --niter 50 --niter_decay 50 --preprocess_mode resize --batchSize 16 --tensorboard_log --use_degradation_v3 --continue_train 
