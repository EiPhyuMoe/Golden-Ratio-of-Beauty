import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
import cv2 as cv2
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import GoldenFace
from numpy import expand_dims
from mtcnn import MTCNN
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from PIL import Image
import time
model = load_model('trained_model_ep15.h5')
st.markdown("<h1 style='text-align: center'>Golden Ratio of Beauty</h1>",unsafe_allow_html=True)
st.info("In this system, we will describe how you are beautiful and we will calculate how your face is symmetric with golden ratio.")



def crop(img): # cropping image to take face only
    # x, y, width, height = result['box']
    s=1.2
    height=img.shape[0]
    width=img.shape[1]
    detector = MTCNN()
    data=detector.detect_faces(img)
    if data==[]:
        return False, None
    else:
        for i, faces in enumerate(data): # iterate through all the faces found
            box=faces['box']  # get the box for each face
            biggest=0
            area = box[2] * box[3]
            if area>biggest:
                biggest=area
                bbox=box
            x,y,w,h=bbox
            xn=int(x +w/2)-int(w * s/2)
            yn=int(y+h/2)- int(h * s/2)
            xen=int(x +w/2) + int(w * s/2)
            yen=int(y+h/2) + int(h * s/2)
            bbox[0]= 0 if bbox[0]<0 else bbox[0]
            xn=0 if xn<0 else xn
            yn=0 if yn<0 else yn
            xen= width if xen>width else xen
            yen= height if yen>height else yen
            img=img[yn:yen, xn:xen]
            return True, img


def main():
    uploaded_file = st.file_uploader(label='Pick an image to test',type =["JPEG","JPG","PNG"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_array = np.array(image)


        #st.image(image,width = 300)
    result = st.button('Run on image')
    if result:
        st.write('Calculating results...')
        my_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.1)
            my_bar.progress(percent_complete + 1)
        status, img_array = crop(image_array) #Uploaded Image Preprocessing
        img_size = (224,224)
        img=cv2.resize(img_array,img_size)
        class_df=pd.read_csv("C:/Users/USER/Desktop/Golden/class_dict.csv")
        scale=class_df['scale by'].iloc[0]
        samples = expand_dims(img, 0)
        datagen = ImageDataGenerator( rescale = 1.0/255.,
                                      rotation_range=0.2,
                                      width_shift_range=0.2,
                                      height_shift_range=0.2,
                                      zoom_range = 0.2,
                                      horizontal_flip=True,
                            )

        it = datagen.flow(samples)
        pred_model = np.squeeze(model.predict(it)) #Beauty Prediction
        index = tf.round(pred_model)
        class_name= class_df['class'].iloc[int(index)]
        #gold = golden('C:/Users/USER/Downloads/lisa1') #Golden Raio Calculation
        st.write(f'She is predicted as being {class_name}')
        st.image(image,caption = class_name,width = 300)
        #st.write(gold)
        genre = st.radio(
        "Do you agree to save your photo for our dataset update?",
        ('Yes', 'No'),index = 1)

        if genre == 'Yes':
            st.info('Thank you so much!!!')

        if genre == 'No':
            st.info('Thank for your time!!!')

     #for camera input
    camera_image = st.checkbox('Let\'s try with camera input')
    if camera_image:
        img_file_camera = st.camera_input("Take a picture")
        if img_file_camera is not None:
            image = Image.open(img_file_camera)
            image_array = np.array(image)


            #st.image(image,width = 300)
            result = st.button('Check result')
        if result:
            st.write('Calculating results...')
            my_bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.1)
                my_bar.progress(percent_complete +1)
            status, img_array = crop(image_array) #Uploaded Image Preprocessing
            img_size = (224,224)
            img=cv2.resize(img_array,img_size)
            class_df=pd.read_csv("C:/Users/USER/Desktop/Golden/class_dict.csv")
            scale=class_df['scale by'].iloc[0]
            samples = expand_dims(img, 0)
            datagen = ImageDataGenerator( rescale = 1.0/255.,
                                          rotation_range=0.2,
                                          width_shift_range=0.2,
                                          height_shift_range=0.2,
                                          zoom_range = 0.2,
                                          horizontal_flip=True,
                                )

            it = datagen.flow(samples)
            pred_model = np.squeeze(model.predict(it)) #Beauty Prediction
            index = tf.round(pred_model)
            class_name= class_df['class'].iloc[int(index)]
            #gold = golden('C:/Users/USER/Downloads/lisa1') #Golden Raio Calculation
            st.write(f'She is predicted as being {class_name}')
            #st.write(gold)
            st.image(image,caption = class_name,width = 300)
            genre = st.radio(
            "Do you agree to save your photo for our dataset update?",
            ('Yes', 'No'),index=1)

            if genre == 'Yes':
                st.info('Thank you so much!!!')
            if genre == 'No':
                st.info('Thank for your time!!!')


if __name__ == '__main__':
    main()
