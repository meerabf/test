# Packages required for Image Classification
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
#import seaborn as sns
#import plotly.express as px

from pathlib import Path
#from sklearn.model_selection import train_test_split

import tensorflow as tf

#from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.keras.models import load_model

#import cv2
import os

import streamlit as st
from PIL import Image
import requests

import base64


cwd = os.getcwd()
print("CURRENT DIR:",cwd)
# To predict the class the image blongs to 
def predict(image1): 

    model =load_model(f'{cwd}/stream_lit_crack_detetction_git/crack_model.h5')
    
    # Load image 
    # using cv2
    ##img = cv2.imread(image1)
    # using Pillow
    #image = Image.open(image1)
    #img = np.array(image)

    ## using matplotlib
    img = plt.imread(image1)

 
    # preprocess the image
    img = img/255 #rescalinng
    new =img.copy()

    new_image = np.resize(new ,new_shape=(120,120,3))


    # predict the confidence of predicting 
    prediction = model.predict(new_image.reshape(-1,120,120,3))
    #st.write(f"Prediction confidence score:{prediction[0][0]}")

    if (prediction<0.5):
        
        label = "No crack was found"
        
    else:
        
        label = "Crack found with a confidence score of " + str(prediction[0][0]*100)+ "%"
        

    return (label,prediction) 


# Get the image from the url for prediction

def get_image(url):
    img = requests.get(url)
    #st.write("Loading the image.................")
    f"""### Loading the image................. """

    ## status code for debugging
    ## st.write(img.status_code)
    file = open("sample_image.jpg", "wb")
    file.write(img.content)
    file.close()
    img_file_name = "sample_image.jpg"
    return img_file_name

# Get the image uploaded for prediction

def get_image_from_up_load(image_file):
    st.write("Loading the image.................")
    
    cwd = os.getcwd()
    with open(os.path.join(cwd,image_file.name),"wb") as f:
			  	f.write((image_file).getbuffer())
    st.success("File Saved")

    img_file_name = image_file.name
    
    return img_file_name

st.title("Concrete Crack detection using CNN")

st.write("If you wish to detect cracks on surfaces, just upload an image or enter the image url and test it out..")

# Main driver
st.sidebar.image(f"{cwd}/stream_lit_crack_detetction_git/crack_detetctive1.jpg", use_column_width=True)
menu = ["Image","Imageurl"]
choice = st.sidebar.selectbox("Select the input data Source",menu)

if choice == "Image":
	
    image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])
    if image_file:

        image = get_image_from_up_load(image_file)
        file_details = {"filename":image_file.name, "filetype":image_file.type,"filesize":image_file.size}
        st.write(file_details)
        st.image(image_file)
        classify = st.button("Find Cracks")

        if classify:
            st.write("")
            st.write(":construction: Please wait while the model is finding cracks. :construction:")

            ## gif from url 
            #st.markdown("![Alt Text](https://miro.medium.com/max/1400/1*BIpRgx5FsEMhr1k2EqBKFg.gif)")
            


            label,prediction = predict(image)


            if label == "Crack found with a confidence score of " + str(prediction[0][0]*100)+ "%":        
                ### gif from local file
                file_ = open(f"{cwd}/stream_lit_crack_detetction_git/crack.gif", "rb")
                contents = file_.read()
                data_url = base64.b64encode(contents).decode("utf-8")
                file_.close()
                st.markdown(f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',unsafe_allow_html=True,)

                #st.write(label)
                f"""### {label}"""
            elif label == "No crack was found":
                ### gif from local file
                file_ = open(f"{cwd}/stream_lit_crack_detetction_git/no_crack.gif", "rb")
                contents = file_.read()
                data_url = base64.b64encode(contents).decode("utf-8")
                file_.close()
                st.markdown(f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',unsafe_allow_html=True,)

                # st.write(label)
                f"""### {label}"""
    else:
        st.write(":point_up_2::skin-tone-2: :point_up_2::skin-tone-2:")


elif choice == "Imageurl":
	
    url = st.text_input("Enter Image Url:")
    if url:

        image = get_image(url)
        st.image(image)
        classify = st.button("Find Cracks")
        if classify:
            st.write("")
            st.write(":construction: Please wait while the model is finding cracks. :construction:")
            label,prediction = predict(image)

            if label == "Crack found with a confidence score of " + str(prediction[0][0]*100)+ "%":        
                ### gif from local file
                file_ = open(f"{cwd}/stream_lit_crack_detetction_git/crack.gif", "rb")
                contents = file_.read()
                data_url = base64.b64encode(contents).decode("utf-8")
                file_.close()
                st.markdown(f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',unsafe_allow_html=True,)
                f"""### {label}"""
                #st.write(label)

                
            elif label == "No crack was found":
                ### gif from local file
                file_ = open(f"{cwd}/stream_lit_crack_detetction_git/no_crack.gif", "rb")
                contents = file_.read()
                data_url = base64.b64encode(contents).decode("utf-8")
                file_.close()
                st.markdown(f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',unsafe_allow_html=True,)
                f"""### {label}"""
                ##st.write(label)

            ##st.markdown("![Alt Text](https://miro.medium.com/max/1400/1*BIpRgx5FsEMhr1k2EqBKFg.gif)")
            ##label = predict(image)        
        ## example used for testing https://github.com/priya-dwivedi/Deep-Learning/blob/master/crack_detection/real_images/concrete_crack1.jpg?raw=true

    else:
        st.write(":point_up_2::skin-tone-2: :point_up_2::skin-tone-2:")