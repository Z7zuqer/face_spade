"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

### To test, give the two folders, which are 256*256 faces and corresponding labels. The save format should be the same with CelebA-HQ
### Remember to use --tensorboard_log, the results will be writed to the [training_folder/test_results]

###


import os
from collections import OrderedDict

import data
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from util.visualizer import Visualizer
from util import html
import torchvision.utils as vutils

opt = TestOptions().parse()

dataloader = data.create_dataloader(opt)

model = Pix2PixModel(opt)
model.eval()

visualizer = Visualizer(opt)

# # create a webpage that summarizes the all results
# web_dir = os.path.join(opt.results_dir, opt.name,
#                        '%s_%s' % (opt.phase, opt.which_epoch))
# webpage = html.HTML(web_dir,
#                     'Experiment = %s, Phase = %s, Epoch = %s' %
#                     (opt.name, opt.phase, opt.which_epoch))

# test

single_save_url=os.path.join(opt.checkpoints_dir, opt.name, opt.results_dir,'each_img')


if not os.path.exists(single_save_url):
    os.makedirs(single_save_url)


for i, data_i in enumerate(dataloader):
    if i * opt.batchSize >= opt.how_many:
        break

    generated = model(data_i, mode='inference')

    img_path = data_i['path']

    visuals = OrderedDict([('input_label', data_i['label']),
                            ('input_image', data_i['image']),
                            ('synthesized_image', generated)])

    visualizer.display_current_results(visuals,None,i)


    ### Also save the single output image


    for b in range(generated.shape[0]):

        img_name=img_path[b].split('/')[-1]
        save_img_url=os.path.join(single_save_url,img_name)

        print('save image... %s' % save_img_url)

        vutils.save_image((generated[b]+1)/2,save_img_url)






    # for b in range(generated.shape[0]):
    #     print('process image... %s' % img_path[b])
    #     visuals = OrderedDict([('input_label', data_i['label'][b]),
    #                            ('synthesized_image', generated[b])])
    #     visualizer.save_images(webpage, visuals, img_path[b:b + 1])

# webpage.save()