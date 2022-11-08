import streamlit as st
import argparse
import io
import os
from PIL import Image
import cv2
import numpy as np
import torch
import time
import re
from io import BytesIO
import easyocr
from functions import *

st.set_page_config(
    page_title="Auto NPR",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="expanded",
)


top_image = Image.open('static/banner_top.png')
bottom_image = Image.open('static/banner_bottom.png')
main_image = Image.open('static/main_banner.png')

upload_path = "uploads/"
download_path = "downloads/"
# initiating the model
model =  torch.hub.load('./yolov7-main', 'custom', source ='local', path_or_model='best.pt',force_reload=True)
classes = model.names

st.image(main_image,use_column_width='auto')
st.title(' Automatic Number Plate Recognition 🚘🚙')
st.sidebar.image(top_image,use_column_width='auto')
st.sidebar.header('Input 🛠')
selected_type = st.sidebar.selectbox('Please select an activity type 🚀', ["Upload Image","Upload Video", "Live Video Feed"])
st.sidebar.image(bottom_image,use_column_width='auto')

# Code for images as input
if selected_type == "Upload Image":
    st.info('✨ Supports all popular image formats 📷 - PNG, JPG, BMP 😉')
    uploaded_file = st.file_uploader("Upload Image of car's number plate 🚓", type=["png","jpg","bmp","jpeg"])

    if uploaded_file is not None:
        with open(os.path.join(upload_path,uploaded_file.name),"wb") as f:
            f.write((uploaded_file).getbuffer())
        with st.spinner(f"Working... 💫"):
            uploaded_image = os.path.abspath(os.path.join(upload_path,uploaded_file.name))
            downloaded_image = os.path.abspath(os.path.join(download_path,str("output_"+uploaded_file.name)))

            with open(uploaded_image,'rb') as imge:
                img_bytes = imge.read()
            
            img = Image.open(io.BytesIO(img_bytes))

            frame =cv2.imread(uploaded_image)

            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        
            result = detectx(frame, model = model) # Detects the position of the license plate

            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

            frame = plot_boxes((result[0],result[1]), frame,classes = classes)
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            img_base64 = Image.fromarray(frame)

            img_base64.save(downloaded_image, format="JPEG")
            
            final_image = Image.open(downloaded_image)
            print("Opening ",final_image)
            st.markdown("---")
            st.image(final_image, caption='This is how your final image looks like 😉')
            
            with open(downloaded_image, "rb") as file:
                if uploaded_file.name.endswith('.jpg') or uploaded_file.name.endswith('.JPG'):
                    if st.download_button(
                                            label="Download Output Image 📷",
                                            data=file,
                                            file_name=str("output_"+uploaded_file.name),
                                            mime='image/jpg'
                                            ):
                        download_success()
                if uploaded_file.name.endswith('.jpeg') or uploaded_file.name.endswith('.JPEG'):
                    if st.download_button(
                                            label="Download Output Image 📷",
                                            data=file,
                                            file_name=str("output_"+uploaded_file.name),
                                            mime='image/jpeg'
                                            ):
                        download_success()

                if uploaded_file.name.endswith('.png') or uploaded_file.name.endswith('.PNG'):
                    if st.download_button(
                                            label="Download Output Image 📷",
                                            data=file,
                                            file_name=str("output_"+uploaded_file.name),
                                            mime='image/png'
                                            ):
                        download_success()

                if uploaded_file.name.endswith('.bmp') or uploaded_file.name.endswith('.BMP'):
                    if st.download_button(
                                            label="Download Output Image 📷",
                                            data=file,
                                            file_name=str("output_"+uploaded_file.name),
                                            mime='image/bmp'
                                            ):
                        download_success()
    else:
        st.warning('⚠ Please upload your Image 😯')

# Code for Live Feed as input
elif selected_type == "Live Video Feed":
    st.info('✨ The Live Feed from Web-Camera will take some time to load up 🎦')
    live_feed = st.checkbox('Start Web-Camera ✅')
    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(0)

    if live_feed:
        while(cap.isOpened()):
            success, frame = cap.read()
            if success == True:
                ret,buffer=cv2.imencode('.jpg',frame)

                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        
                result = detectx(frame, model = model)

                frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

                frame = plot_boxes((result[0],result[1]), frame,classes = classes)

                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            else:
                break
            frame = cv2.imencode('.jpg', frame)[1].tobytes()
            FRAME_WINDOW.image(frame)
    else:
        cap.release()
        st.warning('⚠ The Web-Camera is currently disabled. 😯')

# Code for Video as input
elif selected_type == "Upload Video":

    st.info('✨ Upload a MP4 file ')
    uploaded_file = st.file_uploader("Upload video of car's number plate 🚓", type=["mp4"])
    FRAME_WINDOW = st.image([])

    if uploaded_file is not None:
        with open(os.path.join(upload_path,uploaded_file.name),"wb") as f:
            f.write((uploaded_file).getbuffer())
        with st.spinner(f"Working... 💫"):
            uploaded_video = os.path.abspath(os.path.join(upload_path,uploaded_file.name))
            downloaded_video = os.path.abspath(os.path.join(download_path,str("output_"+uploaded_file.name)))
        
        cap = cv2.VideoCapture(uploaded_video)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*'mp4v') ##(*'XVID')
        out = cv2.VideoWriter(downloaded_video, codec, fps, (width, height))
        
        while(cap.isOpened()):
            success, frame = cap.read()
            if success == True:
                ret,buffer=cv2.imencode('.jpg',frame)
                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        
                result = detectx(frame, model = model)

                frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

                frame = plot_boxes((result[0],result[1]), frame,classes = classes)
                out.write(frame)
 
            else:
                break

            frame = cv2.imencode('.jpg', frame)[1].tobytes()
            FRAME_WINDOW.image(frame)
        out.release()
        cap.release()
        with open(downloaded_video, "rb") as file:
            if uploaded_file.name.endswith('.mp4') or uploaded_file.name.endswith('.mp4'):
                if st.download_button(
                                        label="Download Output video 📷",
                                        data=file,
                                        file_name=str("output_"+uploaded_file.name),
                                        mime='video/mp4'
                                        ):
                    download_success()
        
        
        
