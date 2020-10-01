import os.path
import io
import zipfile
# from data.base_dataset import BaseDataset, get_params, get_transform, normalize
# from data.image_folder import make_dataset
from PIL import Image
import cv2
import torchvision.transforms as transforms
import numpy as np
import random
from io import BytesIO
# from albumentations import MedianBlur,MotionBlur
# from motionblur.motionblur import Kernel
from scipy import ndimage, misc

def pil_to_np(img_PIL):
    '''Converts image in PIL format to np.array.

    From W x H x C [0...255] to C x W x H [0..1]
    '''
    ar = np.array(img_PIL)

    if len(ar.shape) == 3:
        ar = ar.transpose(2, 0, 1)
    else:
        ar = ar[None, ...]

    return ar.astype(np.float32) / 255.


def np_to_pil(img_np):
    '''Converts image in np.array format to PIL image.

    From C x W x H [0..1] to  W x H x C [0...255]
    '''
    ar = np.clip(img_np * 255, 0, 255).astype(np.uint8)

    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        ar = ar.transpose(1, 2, 0)

    return Image.fromarray(ar)


def synthesize_salt_pepper(image,amount,salt_vs_pepper):

    ## Give PIL, return the noisy PIL

    img_pil=pil_to_np(image)

    out = img_pil.copy()
    p = amount
    q = salt_vs_pepper
    flipped = np.random.choice([True, False], size=img_pil.shape,
                               p=[p, 1 - p])
    salted = np.random.choice([True, False], size=img_pil.shape,
                              p=[q, 1 - q])
    peppered = ~salted
    out[flipped & salted] = 1
    out[flipped & peppered] = 0.
    noisy = np.clip(out, 0, 1).astype(np.float32)


    return np_to_pil(noisy)

def synthesize_gaussian(image,std_l,std_r):

    ## Give PIL, return the noisy PIL

    img_pil=pil_to_np(image)

    mean=0
    std=random.uniform(std_l/255.,std_r/255.)
    gauss=np.random.normal(loc=mean,scale=std,size=img_pil.shape)
    noisy=img_pil+gauss
    noisy=np.clip(noisy,0,1).astype(np.float32)

    return np_to_pil(noisy)

def median_blur_ndimage(img):

    filter_size=random.randint(3,17)
    results=ndimage.median_filter(np.array(img),size=(filter_size,filter_size,1))

    return Image.fromarray(results.astype(np.uint8))


# def meadian_blur(img):

#     x=np.array(img)

#     MB=MedianBlur(blur_limit=(3,17), p=1.0)
#     ret=MB(image=x)['image']

#     #print(ret.shape)
#     return Image.fromarray(ret.astype(np.uint8))


# def motion_blur_albumentations(img):

#     x=np.array(img)

#     MB=MotionBlur(blur_limit=(3,26), p=1.0)
#     ret=MB(image=x)['image']

#     #print(ret.shape)
#     return Image.fromarray(ret.astype(np.uint8))


# def random_motion_blur(img):

#     kernel = Kernel(size=(3, 3), intensity=0.8)

#     ret = kernel.applyTo(img)


#     return ret

def synthesize_speckle(image,std_l,std_r):

    ## Give PIL, return the noisy PIL

    img_pil=pil_to_np(image)

    mean=0
    std=random.uniform(std_l/255.,std_r/255.)
    gauss=np.random.normal(loc=mean,scale=std,size=img_pil.shape)
    noisy=img_pil+gauss*img_pil
    noisy=np.clip(noisy,0,1).astype(np.float32)

    return np_to_pil(noisy)


def synthesize_low_resolution(img):
    w,h=img.size

    new_w=random.randint(int(w/4),w)
    new_h=random.randint(int(h/4),h)

    img=img.resize((new_w,new_h),Image.BICUBIC)

    if random.uniform(0,1)<0.5:
        img=img.resize((w,h),Image.NEAREST)
    else:
        img = img.resize((w, h), Image.BILINEAR)

    return img

def convertToJpeg(im,quality):
    with BytesIO() as f:
        im.save(f, format='JPEG',quality=quality)
        f.seek(0)
        return Image.open(f).convert('RGB')


def blur_image_v2(img):


    x=np.array(img)
    kernel_size_candidate=[(3,3),(5,5),(7,7)]
    kernel_size=random.sample(kernel_size_candidate,1)[0]
    std=random.uniform(1.,5.)

    #print("The gaussian kernel size: (%d,%d) std: %.2f"%(kernel_size[0],kernel_size[1],std))
    blur=cv2.GaussianBlur(x,kernel_size,std)

    return Image.fromarray(blur.astype(np.uint8))

def online_add_degradation(img):

    task_id=np.random.permutation(4)

    for x in task_id:
        if x==0 and random.uniform(0,1)<0.7:
            img = blur_image_v2(img)
        if x==1 and random.uniform(0,1)<0.7:
            flag = random.choice([1, 2, 3])
            if flag == 1:
                img = synthesize_gaussian(img, 5, 50)
            if flag == 2:
                img = synthesize_speckle(img, 5, 50)
            if flag == 3:
                img = synthesize_salt_pepper(img, random.uniform(0, 0.01), random.uniform(0.3, 0.8))
        if x==2 and random.uniform(0,1)<0.7:
            img=synthesize_low_resolution(img)

        if x==3 and random.uniform(0,1)<0.7:
            img=convertToJpeg(img,random.randint(40,100))

    return img


# def online_add_degradation_v2(img):

#     task_id=np.random.permutation(4)

#     for x in task_id:

#         if x==0 and random.uniform(0,1)<0.5:

#             if random.uniform(0,1)<0.5:
#                 img = blur_image_v2(img)
#             else:
#                 img = motion_blur_albumentations(img)
            
#         if x==1 and random.uniform(0,1)<0.7:
#                 img = synthesize_gaussian(img, 5, 50)
#         if x==2 and random.uniform(0,1)<0.7:
#             img=synthesize_low_resolution(img)
#         if x==3 and random.uniform(0,1)<0.7:
#             img=convertToJpeg(img,random.randint(15,100))


#     if random.uniform(0,1)<0.8:
#         img=meadian_blur(img)
    
#     return img


def online_add_degradation_v3(img):  ## Remove motion blur of v2

    task_id=np.random.permutation(4)

    for x in task_id:

        if x==0 and random.uniform(0,1)<0.5:

            img = blur_image_v2(img)
            
        if x==1 and random.uniform(0,1)<0.7:
                img = synthesize_gaussian(img, 5, 50)
        if x==2 and random.uniform(0,1)<0.7:
            img=synthesize_low_resolution(img)
        if x==3 and random.uniform(0,1)<0.7:
            img=convertToJpeg(img,random.randint(15,100))


    if random.uniform(0,1)<0.8:
        img=median_blur_ndimage(img)
    
    return img


if __name__=='__main__':

    image_url='test.jpg'
    img=Image.open(image_url).convert("RGB")
    x=img.resize((256,256),resample=Image.BICUBIC)

    #y=online_add_degradation(x)
    #y=meadian_blur(x)
    #y=motion_blur_albumentations(x)
    #y=random_motion_blur(x)
    y=online_add_degradation_v2(x)
    print(y.size)
    #x.show()
    y.show()
