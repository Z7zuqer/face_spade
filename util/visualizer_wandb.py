"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
import ntpath
import time
from . import util
from . import html
import scipy.misc
try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
import torch
import numpy as np

class Visualizer():
    def __init__(self, opt):
        self.opt = opt
        self.tf_log = opt.isTrain and opt.tf_log

        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name

        if self.opt.isTrain:
            self.log_dir = os.path.join(opt.checkpoints_dir, opt.name, 'logs')
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
        else:
            print("hi :)")
            self.log_dir = os.path.join(opt.checkpoints_dir, opt.name, opt.results_dir)
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)               
            # import tensorflow as tf
            # self.tf = tf
            # self.log_dir = os.path.join(opt.checkpoints_dir, opt.name, 'logs')
            # self.writer = tf.summary.FileWriter(self.log_dir)

        if opt.isTrain:
            self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
            with open(self.log_name, "a") as log_file:
                now = time.strftime("%c")
                log_file.write('================ Training Loss (%s) ================\n' % now)

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch, step):


        all_tensor=[]
            
        for key,tensor in visuals.items():
            if key=='input_label':
                tile = self.opt.batchSize > 1
                t = util.tensor2label(tensor, self.opt.label_nc + 2, tile=tile) ## B*H*W*3 0-255 numpy
                t = np.transpose(t,(0,3,1,2))
                all_tensor.append(torch.tensor(t).float()/255)
            else:
                all_tensor.append((tensor.data.cpu()+1)/2)
            
    

        output=torch.cat(all_tensor,0)
        img_grid=vutils.make_grid(output,nrow=self.opt.batchSize,padding=0,normalize=False)

        if self.opt.isTrain:
            return img_grid
        else:
            # self.writer.add_image('Face_SPADE/test_samples',img_grid,step)
            vutils.save_image(output,os.path.join(self.log_dir,str(step)+'.png'),nrow=self.opt.batchSize,padding=0,normalize=False)
        

        ## convert tensors to numpy arrays
        # visuals = self.convert_visuals_to_numpy(visuals)
                
        # if self.tf_log: # show images in tensorboard output
        #     img_summaries = []
        #     for label, image_numpy in visuals.items():
        #         # Write the image to a string
        #         try:
        #             s = StringIO()
        #         except:
        #             s = BytesIO()
        #         if len(image_numpy.shape) >= 4:
        #             image_numpy = image_numpy[0]
        #         scipy.misc.toimage(image_numpy).save(s, format="jpeg")
        #         # Create an Image object
        #         img_sum = self.tf.Summary.Image(encoded_image_string=s.getvalue(), height=image_numpy.shape[0], width=image_numpy.shape[1])
        #         # Create a Summary value
        #         img_summaries.append(self.tf.Summary.Value(tag=label, image=img_sum))

        #     # Create and write Summary
        #     summary = self.tf.Summary(value=img_summaries)
        #     self.writer.add_summary(summary, step)


    # errors: dictionary of error labels and values
    # def plot_current_errors(self, errors, step):
    #     if self.tf_log:
    #         for tag, value in errors.items():
    #             value = value.mean().float()
    #             summary = self.tf.Summary(value=[self.tf.Summary.Value(tag=tag, simple_value=value)])
    #             self.writer.add_summary(summary, step)
        
    #     if self.tensorboard_log:

    #     #    self.writer.add_scalar('G',errors['GAN'].item(),step)
    #         self.writer.add_scalar('Loss/GAN_Feat',errors['GAN_Feat'].mean().float(),step)
    #         self.writer.add_scalar('Loss/VGG',errors['VGG'].mean().float(),step)
    #     #    self.writer.add_scalar('D',(errors['D_Fake'].item()+errors['D_real'].item())/2,step)
    #         self.writer.add_scalars('Loss/GAN',{'G':errors['GAN'].mean().float(),
    #                                             'D':(errors['D_Fake'].mean().float()+errors['D_real'].mean().float())/2},
    #                                             step)


    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, errors, t):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        for k, v in errors.items():
            #print(v)
            #if v != 0:
            v = v.mean().float()
            message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    def convert_visuals_to_numpy(self, visuals):
        for key, t in visuals.items():
            tile = self.opt.batchSize > 8
            if 'input_label' == key:
                t = util.tensor2label(t, self.opt.label_nc + 2, tile=tile) ## B*H*W*C 0-255 numpy
            else:
                t = util.tensor2im(t, tile=tile)
            visuals[key] = t
        return visuals

    # save image to the disk
    def save_images(self, webpage, visuals, image_path):        
        visuals = self.convert_visuals_to_numpy(visuals)        
        
        image_dir = webpage.get_image_dir()
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            image_name = os.path.join(label, '%s.png' % (name))
            save_path = os.path.join(image_dir, image_name)
            util.save_image(image_numpy, save_path, create_dir=True)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        webpage.add_images(ims, txts, links, width=self.win_size)
