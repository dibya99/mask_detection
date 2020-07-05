import tensorflow as tf
import os
from os import getcwd
from os import listdir
import cv2
import numpy as np
from pathlib import Path
import numpy as np

curr_dir=str(Path.cwd())
main_dir=curr_dir[0:curr_dir.rfind("/")]

def load(path):
    print('Hi')
    model=tf.keras.models.load_model(path)
    print('Bye')
    model.summary()
    return model

def run_on_camera():
    model_path=main_dir+'/saved_models/mask_detector.h5'
    model=load(model_path)

    face_classifier=cv2.CascadeClassifier(main_dir+'/util/haarcascade_frontalface_default.xml')

    labels_dict={0:'without_mask',1:'with_mask'}
    color_dict={0:(0,0,255),1:(0,255,0)}

    size = 4
    webcam = cv2.VideoCapture(0) #Use camera 0
    while True:
        (rval, im) = webcam.read()
        im=cv2.flip(im,1,1) #Flip to act as a mirror

        # Resize the image to speed up detection
        mini = cv2.resize(im, (im.shape[1] // size, im.shape[0] // size))

        # detect MultiScale / faces
        faces = face_classifier.detectMultiScale(im)

    # Draw rectangles around each face
        for f in faces:
            (x, y, w, h) = [v * 1 for v in f] #Scale the shapesize backup
        #Save just the rectangle faces in SubRecFaces
            face_img = im[y:y+h, x:x+w]
            resized=cv2.resize(face_img,(150,150))
            normalized=resized/255.0
            normalized = normalized[:, :, ::-1].copy()
            reshaped=np.reshape(normalized,(1,150,150,3))
            reshaped = np.vstack([reshaped])
            result=model.predict(reshaped)
        #print(result)
            result=result[0][0]

            if result>=0.5:
                label=0
            else:
                label=1
            print(label)
            cv2.rectangle(im,(x,y),(x+w,y+h),color_dict[label],2)
            cv2.rectangle(im,(x,y-40),(x+w,y),color_dict[label],-1)
            cv2.putText(im, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

            # Show the image
        cv2.imshow('LIVE',   im)
        key = cv2.waitKey(10)
    # if Esc key is press then break out of the loop
        if key == 27: #The Esc key
            break
        # Stop video
    webcam.release()

    # Close all started windows
    cv2.destroyAllWindows()

run_on_camera()
