"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import sys
from collections import OrderedDict
from options.train_options import TrainOptions
import data
from util.iter_counter import IterationCounter
from util.visualizer_wandb import Visualizer
from trainers.pix2pix_trainer import Pix2PixTrainer
import warnings
import wandb
# parse options

warnings.filterwarnings('ignore')


opt = TrainOptions().parse()

opt.contain_dontcare_label=False
# print options to help debugging
print(' '.join(sys.argv))


wandb.init(project='Face_SPADE',name=opt.name)


# load the dataset
#dataloader = data.create_dataloader(opt)
dataloader=data.create_dataloader(opt)


# create trainer for our model
trainer = Pix2PixTrainer(opt)

# create tool for counting iterations
iter_counter = IterationCounter(opt, len(dataloader))

# create tool for visualization
visualizer = Visualizer(opt)

for epoch in iter_counter.training_epochs():
    iter_counter.record_epoch_start(epoch)
    for i, data_i in enumerate(dataloader, start=iter_counter.epoch_iter):
        iter_counter.record_one_iteration()

        # Training
        # train generator
        if i % opt.D_steps_per_G == 0:
            trainer.run_generator_one_step(data_i)

        # train discriminator
        trainer.run_discriminator_one_step(data_i)

        # Visualizations
        if iter_counter.needs_printing():
            losses = trainer.get_latest_losses()
            visualizer.print_current_errors(epoch, iter_counter.epoch_iter,
                                            losses, iter_counter.time_per_iter)
            # visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)
            wandb.log({"GAN_Feat":losses['GAN_Feat'].mean().float(),"VGG":losses['VGG'].mean().float(),"GAN_G":losses['GAN'].mean().float(),"GAN_D":(losses['D_Fake'].mean().float()+losses['D_real'].mean().float())/2})

        if iter_counter.needs_displaying():
            visuals = OrderedDict([('input_label', data_i['label']),
                                    ('degraded_image', data_i['degraded_image']),
                                   ('synthesized_image', trainer.get_latest_generated()),
                                   ('real_image', data_i['image'])])
            img_grid=visualizer.display_current_results(visuals, epoch, iter_counter.total_steps_so_far)
            wandb.log({"Training_Samples": wandb.Image(img_grid)})

        if iter_counter.needs_saving():
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, iter_counter.total_steps_so_far))
            trainer.save('latest')
            iter_counter.record_current_iter()

    trainer.update_learning_rate(epoch)
    iter_counter.record_epoch_end()

    if epoch % opt.save_epoch_freq == 0 or \
       epoch == iter_counter.total_epochs:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, iter_counter.total_steps_so_far))
        trainer.save('latest')
        trainer.save(epoch)

print('Training was successfully finished.')