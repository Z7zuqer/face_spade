from data.base_dataset import BaseDataset, get_params, get_transform
from PIL import Image
import util.util as util
import os
from data.degradation import online_add_degradation,online_add_degradation_v3
import torch


class FaceTestDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--no_pairing_check', action='store_true',
                            help='If specified, skip sanity check of correct label-image file pairing')
    #    parser.set_defaults(contain_dontcare_label=False)
    #    parser.set_defaults(no_instance=True)
        return parser

    def initialize(self, opt):
        self.opt = opt

        image_path=os.path.join(opt.dataroot,opt.old_face_folder)
        label_path=os.path.join(opt.dataroot,opt.old_face_label_folder)

        image_list=os.listdir(image_path)
        image_list=sorted(image_list)
        # image_list=image_list[:opt.max_dataset_size]


        self.label_paths = label_path ## Just the root dir
        self.image_paths = image_list ## All the image name


        self.parts=['skin','hair', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g','l_ear', 'r_ear', 'ear_r', 'nose', 'mouth','u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hat']

        size = len(self.image_paths)
        self.dataset_size = size


    def __getitem__(self, index):
        # Label Image
        # label_path = self.label_paths[index]
        # label = Image.open(label_path)

        # params = get_params(self.opt, label.size)
        # transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        # label_tensor = transform_label(label) * 255.0
        # label_tensor[label_tensor == 255] = self.opt.label_nc  # 'unknown' is opt.label_nc



        params=get_params(self.opt,(-1,-1))
        # input image (real images)
        image_name = self.image_paths[index]
        # assert self.paths_match(label_path, image_path), \
        #     "The label_path %s and image_path %s don't match." % \
        #     (label_path, image_path)
        image_path=os.path.join(self.opt.dataroot,self.opt.old_face_folder,image_name)
        image = Image.open(image_path)
        image = image.convert('RGB')



        transform_image = get_transform(self.opt, params)
        image_tensor = transform_image(image)
    #    degraded_image_tensor = transform_image(degraded_image)
        


        ## From the image name to search corresponding parsing mask

        img_name=image_name[:-4]
    #    label_folder=int(img_id/2000)
        transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        full_label=[]

        cnt=0

        for each_part in self.parts:
            part_name=img_name+'_'+each_part+'.png'
            part_url=os.path.join(self.label_paths,part_name)

            if os.path.exists(part_url):
                label=Image.open(part_url).convert("RGB")
                label_tensor=transform_label(label) ## 3 channels and pixel [0,1]
                full_label.append(label_tensor[0])
            else:
                current_part=torch.zeros((self.opt.load_size,self.opt.load_size))
                full_label.append(current_part)
                cnt+=1

        # if cnt>8:
        #     print("Maybe the part is not searched well")
        #     print(image_path)

        full_label_tensor=torch.stack(full_label,0)



    
        # # if using instance maps
        # if self.opt.no_instance:
        #     instance_tensor = 0
        # else:
        #     instance_path = self.instance_paths[index]
        #     instance = Image.open(instance_path)
        #     if instance.mode == 'L':
        #         instance_tensor = transform_label(instance) * 255
        #         instance_tensor = instance_tensor.long()
        #     else:
        #         instance_tensor = transform_label(instance)

        input_dict = {'label': full_label_tensor,
                    #   'instance': instance_tensor,
                        # 'degraded_image': degraded_image_tensor,
                      'image': image_tensor,
                      'path': image_path,
                      }


        return input_dict


    def __len__(self):
        return self.dataset_size



class FaceDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--no_pairing_check', action='store_true',
                            help='If specified, skip sanity check of correct label-image file pairing')
    #    parser.set_defaults(contain_dontcare_label=False)
    #    parser.set_defaults(no_instance=True)
        return parser

    def initialize(self, opt):
        self.opt = opt

        image_path=os.path.join(opt.dataroot,'CelebA-HQ-img')
        label_path=os.path.join(opt.dataroot,'CelebAMask-HQ-mask-anno')

        image_list=os.listdir(image_path)
        image_list=sorted(image_list)
        image_list=image_list[:opt.max_dataset_size]


        self.label_paths = label_path ## Just the root dir
        self.image_paths = image_list ## All the image name


        self.parts=['skin','hair', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g','l_ear', 'r_ear', 'ear_r', 'nose', 'mouth','u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hat']

        size = len(self.image_paths)
        self.dataset_size = size


    def __getitem__(self, index):
        # Label Image
        # label_path = self.label_paths[index]
        # label = Image.open(label_path)

        # params = get_params(self.opt, label.size)
        # transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        # label_tensor = transform_label(label) * 255.0
        # label_tensor[label_tensor == 255] = self.opt.label_nc  # 'unknown' is opt.label_nc



        params=get_params(self.opt,(-1,-1))
        # input image (real images)
        image_name = self.image_paths[index]
        # assert self.paths_match(label_path, image_path), \
        #     "The label_path %s and image_path %s don't match." % \
        #     (label_path, image_path)
        image_path=os.path.join(self.opt.dataroot,'CelebA-HQ-img',image_name)
        image = Image.open(image_path)
        image = image.convert('RGB')
        image = image.resize((self.opt.load_size,self.opt.load_size),Image.BICUBIC)

        # TODO: Add degradation to the down-sampled image

        
        if self.opt.use_degradation_v2:
            degraded_image=online_add_degradation_v2(image)

        elif self.opt.use_degradation_v3:
            degraded_image=online_add_degradation_v3(image)
        else:
            degraded_image=online_add_degradation(image)
        #


        transform_image = get_transform(self.opt, params)
        image_tensor = transform_image(image)
        degraded_image_tensor = transform_image(degraded_image)



        ## From the image name to search corresponding parsing mask

        img_id=int(image_name.split('.')[-2])
        label_folder=int(img_id/2000)
        transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        full_label=[]

        cnt=0

        for each_part in self.parts:
            part_name=str(img_id).zfill(5)+'_'+each_part+'.png'
            part_url=os.path.join(self.label_paths,str(label_folder),part_name)

            if os.path.exists(part_url):
                label=Image.open(part_url).convert("RGB")
                label_tensor=transform_label(label) ## 3 channels and pixel [0,1]
                full_label.append(label_tensor[0])
            else:
                current_part=torch.zeros((self.opt.load_size,self.opt.load_size))
                full_label.append(current_part)
                cnt+=1

        # if cnt>8:
        #     print("Maybe the part is not searched well")
        #     print(image_path)

        full_label_tensor=torch.stack(full_label,0)



    
        # # if using instance maps
        # if self.opt.no_instance:
        #     instance_tensor = 0
        # else:
        #     instance_path = self.instance_paths[index]
        #     instance = Image.open(instance_path)
        #     if instance.mode == 'L':
        #         instance_tensor = transform_label(instance) * 255
        #         instance_tensor = instance_tensor.long()
        #     else:
        #         instance_tensor = transform_label(instance)

        input_dict = {'label': full_label_tensor,
                    #   'instance': instance_tensor,
                        'degraded_image': degraded_image_tensor,
                      'image': image_tensor,
                      'path': image_path,
                      }


        return input_dict


    def __len__(self):
        return self.dataset_size