import streamlit as st
import cv2
import matplotlib.pyplot as plt
import numpy as np
from urllib.request import urlopen
from PIL import Image
import random
from matplotlib.pyplot import figure
import glob
import pytesseract
import re
import easyocr
import math
from scipy import ndimage


# function for finding number plate rotation angle
def find_alignment_angle(image):
    median_angle=0
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_edges = cv2.Canny(img_gray, 100, 100, apertureSize=3)
    lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=5)
    
    angles = []
    try:
        
        for [[x1, y1, x2, y2]] in lines:
            cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 3)
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            angles.append(angle)
            
        median_angle = np.median(angles)
        print(median_angle)
    except:
        pass
    return median_angle
# convert list into string
def lst_str(lst):
    str1 = "" 
    for ele in lst: 
        str1 += ele  
    return str1


# main code
reader=easyocr.Reader(['en'])

#Load Yolo v4 on custom trained weights
weights='yolov4_train_last.weights'
configuration='yolov4_train.cfg'
yolov4 = cv2.dnn.readNet(weights,configuration)

classes=['Vehicle registration plate']


st.title("Vehicle number plate recognition")
image_file=st.file_uploader("Choose a image file" , type=['jpg','png','jpeg'])

    
if image_file is not None:
    img=np.asarray(bytearray(image_file.read()),dtype=np.uint8)
    img=cv2.imdecode(img,1)
    
    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 1/255 , (416 , 416) ,(0,0,0) , swapRB=True , crop=False)
    yolov4.setInput(blob)
    output_layer_name=yolov4.getUnconnectedOutLayersNames()
    layeroutput=yolov4.forward(output_layer_name)
    
    
    boxes=[]
    confidences=[]
    class_ids=[]
    accu_score=0.6
    box_width=2
    text_size=2
    for output in layeroutput:
        for detection in output:
            score=detection[5:]
            class_id=np.argmax(detection[5:])
            confidence=score[class_id]
            if confidence >accu_score:
                center_x=int(detection[0]*width)
                center_y=int(detection[1]*height)
                w=int(detection[2]*width)
                h=int(detection[3]*height)
      
                x=int(center_x-w/2)
                y=int(center_y-h/2)
      
                boxes.append([x,y,w,h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)


    font = cv2.FONT_HERSHEY_PLAIN
    colors=[random.uniform(0,255) for i in range(len(boxes))]
    crop_img=[]
    img1=img.copy()
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
    
            label = str(classes[class_ids[i]])
            color = colors[i]
            confi=str(round(confidences[i],2))
#             rearrange box width

            if height*width>2000000:
                box_width=4
                text_size=4
            crop_img.append(img[y:y+h, x:x+w])
            cv2.rectangle(img1, (x,y), (x + w, y + h), color, box_width)
            
            (text_width, text_height) = cv2.getTextSize(label+ " "+confi, font, 2, text_size)[0]
            cv2.rectangle(img1, (x,y-25), (x + text_width, y + text_height), color, cv2.FILLED)
            cv2.putText(img1, label+ " "+confi, (x, y+2), font, 2, (255,255,255),text_size)
          
            
    
    
#     show image 
    st.image(img1,channels="RGB")
    
    
    
    for i in crop_img:
        st.text(f'Vehicle registration plate :- {lst_str(reader.readtext(i, detail = 0 , paragraph=True))}')
        st.image(i,channels="RGB") 
        
        angle=find_alignment_angle(i)
        if abs(angle)>8:
            temp_img=ndimage.rotate(i, angle)
            st.text(f'Vehicle registration plate :- {lst_str(reader.readtext(temp_img, detail = 0 , paragraph=True))}')
            st.image(temp_img,channels="RGB")
