# Face_SPADE


### How to use?



2020.4.15

    Implement the basic functions.

    python train.py --name [experiment_name] --dataset_mode custom --label_dir [path_to_labels] --image_dir [path_to_images] --label_nc [num_labels]


    python train_face.py --name debug --load_size 256 --gpu_ids 0 --label_nc 18 --dataroot /home/ziyuwan/datasets/CelebAMask-HQ --no_instance --max_dataset_size 29000 --nThreads 2 --save_epoch_freq 1 --niter 50 --niter_decay 50 --preprocess_mode resize


    /home/ziyuwan/Face_SPADE/checkpoints/debug/web


2020.4.18 

    [Transfer the code from my gpu to facednn]


    python train_face.py --name debug --load_size 256 --gpu_ids 0,1 --label_nc 18 --dataroot /home/jingliao/ziyuwan/workspace/dataset/CelebAMask-HQ --no_instance --max_dataset_size 29000 --nThreads 4 --save_epoch_freq 1 --niter 50 --niter_decay 50 --preprocess_mode resize --batchSize 8 --tensorboard_log 


    python train_face.py --name basic_setting --load_size 256 --gpu_ids 2,3 --label_nc 18 --dataroot /home/jingliao/ziyuwan/workspace/dataset/CelebAMask-HQ --no_instance --max_dataset_size 29000 --nThreads 4 --save_epoch_freq 1 --niter 50 --niter_decay 50 --preprocess_mode resize --batchSize 8 --tensorboard_log 



2020.4.20 

    python test_face.py --name debug --load_size 256 --gpu_ids 0 --label_nc 18 --dataroot /home/jingliao/ziyuwan/workspace/dataset/ --no_instance --nThreads 4 --preprocess_mode resize --batchSize 4 --tensorboard_log --old_face_folder --old_face_label_folder 

    python train_face.py --name degradation_v2 --load_size 256 --gpu_ids 4,5 --label_nc 18 --dataroot /home/jingliao/ziyuwan/workspace/dataset/CelebAMask-HQ --no_instance --max_dataset_size 29000 --nThreads 4 --save_epoch_freq 1 --niter 50 --niter_decay 50 --preprocess_mode resize --batchSize 8 --tensorboard_log --use_degradation_v2


2020 4.27

    python train_face.py --name degradation_v2 --load_size 256 --gpu_ids 4,5 --label_nc 18 --dataroot /home/jingliao/ziyuwan/workspace/dataset/CelebAMask-HQ --no_instance --max_dataset_size 29000 --nThreads 4 --save_epoch_freq 1 --niter 50 --niter_decay 50 --preprocess_mode resize --batchSize 8 --tensorboard_log --use_degradation_v2 --continue_train



2020.4.30

    python test_face.py --old_face_folder /home/jingliao/ziyuwan/face_Old --old_face_label_folder /home/jingliao/ziyuwan/face_Old_mask --tensorboard_log --name degradation_v2 --gpu_ids 7 --load_size 256 --label_nc 18 --no_instance --preprocess_mode resize --batchSize 4


2020.5.1


    python test_face.py --old_face_folder /home/jingliao/ziyuwan/bill_detected_face --old_face_label_folder /home/jingliao/ziyuwan/bill_face_mask --tensorboard_log --name degradation_v2 --gpu_ids 2 --load_size 256 --label_nc 18 --no_instance --preprocess_mode resize --batchSize 4 --results_dir ./bill_face/


    python test_face.py --old_face_folder /home/jingliao/ziyuwan/bill_detected_face_new_restored --old_face_label_folder /home/jingliao/ziyuwan/result_now/bill_restored_face_mask --tensorboard_log --name degradation_v2 --gpu_ids 2 --load_size 256 --label_nc 18 --no_instance --preprocess_mode resize --batchSize 4 --results_dir ./bill_restored_face/

    python test_face.py --old_face_folder /home/jingliao/ziyuwan/celebrities_detected_face_new_restored --old_face_label_folder /home/jingliao/ziyuwan/result_now/celebrities_restored_face_mask --tensorboard_log --name degradation_v2 --gpu_ids 2 --load_size 256 --label_nc 18 --no_instance --preprocess_mode resize --batchSize 4 --results_dir ./celebrities_restored_face/



2020.6.10

[Train the face spade model using degradation v3]

    python train_face.py --name degradation_v3 --load_size 256 --gpu_ids 3,4 --label_nc 18 --dataroot /home/jingliao/ziyuwan/workspace/dataset/CelebAMask-HQ --no_instance --max_dataset_size 29000 --nThreads 4 --save_epoch_freq 1 --niter 50 --niter_decay 50 --preprocess_mode resize --batchSize 8 --tensorboard_log --use_degradation_v3


2020.6.11


[GDB debug]

    python train_face.py --name gdb_debug --load_size 256 --gpu_ids 5,6 --label_nc 18 --dataroot /home/jingliao/ziyuwan/workspace/dataset/CelebAMask-HQ --no_instance --max_dataset_size 29000 --nThreads 4 --save_epoch_freq 1 --niter 50 --niter_decay 50 --preprocess_mode resize --batchSize 8 --tensorboard_log --use_degradation_v2


    python train_face_wandb.py --name debug --load_size 256 --gpu_ids 2,7 --label_nc 18 --dataroot /home/jingliao/ziyuwan/workspace/dataset/CelebAMask-HQ --no_instance --max_dataset_size 29000 --nThreads 4 --save_epoch_freq 1 --niter 50 --niter_decay 50 --preprocess_mode resize --batchSize 8 --tensorboard_log --use_degradation_v3 --continue_train 

    python train_face_wandb.py --name debug --load_size 256 --gpu_ids 2,7 --label_nc 18 --dataroot /home/jingliao/ziyuwan/workspace/dataset/CelebAMask-HQ --no_instance --max_dataset_size 29000 --nThreads 4 --save_epoch_freq 1 --niter 50 --niter_decay 50 --preprocess_mode resize --batchSize 8 --tensorboard_log --use_degradation_v3 --continue_train 




2020.6.13

[Train without Parsing map]

    python train_face_wandb.py --name Setting_4_debug --load_size 256 --gpu_ids 0,1 --label_nc 18 --dataroot /home/jingliao/ziyuwan/workspace/dataset/CelebAMask-HQ-Aligned --no_instance --max_dataset_size 29000 --nThreads 4 --save_epoch_freq 1 --niter 50 --niter_decay 50 --preprocess_mode resize --batchSize 8 --tensorboard_log --use_degradation_v3 --continue_train --no_parsing_map