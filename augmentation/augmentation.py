import cv2
import os
from pathlib import Path
from PIL import Image
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img



def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = load_img(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

def augment(images,prefix,main_dir,datagen):
    for img in images:
        #img=cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        x = img_to_array(img)
        # Reshape the input image
        print(x.shape)
        x = x.reshape((1, ) + x.shape)
        print(x.shape)
        i = 0

        # generate X new augmented images
        for batch in datagen.flow(x, batch_size = 1,
                          save_to_dir =main_dir+'/datasets/augmented/'+prefix+'/',
                          save_prefix =prefix, save_format ='jpg'):
            i += 1
            if i >= 3:
                break


def main():
    datagen = ImageDataGenerator(      rescale=1./255,
                                     #  preprocessing_function=vaibhav,
                                       rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       fill_mode='nearest')
    curr_dir=str(Path.cwd())
    main_dir=curr_dir[0:curr_dir.rfind("/")]

    #for_with_mask_images
    folder_path_1=main_dir+"/datasets/original/with_mask"
    with_mask_images=load_images_from_folder(folder_path_1)
    augment(with_mask_images,"with_mask",main_dir,datagen)

    #for_without_mask_images
    folder_path_2=main_dir+"/datasets/original/without_mask"
    without_mask_images=load_images_from_folder(folder_path_2)
    augment(without_mask_images,"without_mask",main_dir,datagen)

    '''
    img=load_img('/home/dibya/verzeo/Mask_detection/datasets/original/with_mask/0-with-mask.jpg')
    #print("Image shape is: ",img.shape)
    #cv2.imshow("window",img)
    #cv2.waitKey(2000)
    images=[img]
    prefix='demo'
    augment(images,prefix,main_dir,datagen)
    '''
main()
