import cv2
import tensorflow as tf
import os
import numpy as np
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

model=tf.keras.models.load_model('/home/dibya/verzeo/Mask_detection/saved_models/mask_detector.h5')

'''
for filename in os.listdir('/home/dibya/verzeo/Mask_detection/datasets/augmented/with_mask/'):
    #opencv

    img = cv2.imread('/home/dibya/verzeo/Mask_detection/datasets/augmented/with_mask/'+filename)
    resized=cv2.resize(img,(150,150))
    normalized=resized/255.0
    normalized=np.array(normalized)
    normalized = normalized[:, :, ::-1].copy()
    cv2.imshow("window",img)
    cv2.waitKey(200)
    reshaped=np.reshape(normalized,(1,150,150,3))
    reshaped = np.vstack([reshaped])
    result=model.predict(reshaped)
    print(result)

    #ends

    #PIL

    #img=load_img('/home/dibya/verzeo/Mask_detection/datasets/augmented/without_mask/'+filename,target_size=(150,150))
    #normalized=img
    #print(img)

    #ends


    '''
webcam = cv2.VideoCapture(0)
while True:
    ret,im = webcam.read()
    resized=cv2.resize(im,(150,150))
    normalized=resized/255.0
    normalized=np.array(normalized)
    normalized = normalized[:, :, ::-1].copy()
    reshaped=np.reshape(normalized,(1,150,150,3))
    reshaped = np.vstack([reshaped])
    result=model.predict(reshaped)
    print(result[0][0])
    cv2.imshow("LIVE",im)
    cv2.waitKey(10)
#print(result)
    print(result)
