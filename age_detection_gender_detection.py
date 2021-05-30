# -*- coding: utf-8 -*-
"""
Created on Fri May 28 00:31:52 2021

@author: Gundeep Gulati
"""


import cv2
import os
os.chdir(r'C:\Users\Gundeep Gulati\Desktop\Age-and-Gender-Recognition-main\Age-and-Gender-Recognition-main\models')
def Face_detect(net, frame, confidence_threshold=0.7):
    frameOpen = frame.copy()
    frameHeight = frameOpen.shape[0]
    frameWidth = frameOpen.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpen,1.0,(227,227),[124.96,115.97,106.13],swapRB=True,crop=False) #convert image into 4d array 1.0 scale factor,(227,227) image size, True for swap rb(red nd blue), False for crop
    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]
        if confidence>confidence_threshold:
            x1 = int(detections[0,0,i,3]*frameWidth)
            y1 = int(detections[0,0,i,4]*frameHeight)
            x2 = int(detections[0,0,i,5]*frameWidth)
            y2 = int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpen,(x1,y1),(x2,y2),(0,255,0),int(round(frameHeight/150)),8)
        return frameOpen, faceBoxes

faceProto = 'opencv_face_detector.pbtxt'
faceModel = 'opencv_face_detector_uint8.pb'
ageProto = 'age_deploy.prototxt'
ageModel = 'age_net.caffemodel'
genderProto = 'gender_deploy.prototxt'
genderModel = 'gender_net.caffemodel'

genderList = ['Male','Female']
ageList=['(0-2)','(4-6)','(8-12)','(15-20)','(25-32)','(38-43)','(48-53)','(60-100)']

faceNet = cv2.dnn.readNet(faceModel,faceProto)
ageNet = cv2.dnn.readNet(ageModel,ageProto)
genderNet = cv2.dnn.readNet(genderModel,genderProto)

video = cv2.VideoCapture(0)
padding=20

while cv2.waitKey(1)<0:
    hasFrame,Frame=video.read()
    if not hasFrame:
        cv2.waitKey()
        break
        
    resultImg,faceBoxes = Face_detect(faceNet,Frame)
    
    if not faceBoxes:
        print("No face detected")
    for facebox in faceBoxes:
        face=Frame[max(0,facebox[1]-padding):min(facebox[3]+padding,Frame.shape[0]-1),max(0,facebox[0]-padding):min(facebox[2]+padding, Frame.shape[1]-1)]
        blob = cv2.dnn.blobFromImage(face,1.0,(227,227),[124.96,115.97,106.13],swapRB=True,crop=False) #convert image into 4d array 1.0 scale factor,(227,227) image size, True for swap rb(red nd blue), False for crop
        genderNet.setInput(blob)
        genderPreds=genderNet.forward()
        gender=genderList[genderPreds[0].argmax()]
        print(gender)
        
        ageNet.setInput(blob)
        agePreds=ageNet.forward()
        age=ageList[agePreds[0].argmax()]
        print(age)
        cv2.putText(resultImg,f'{gender},{age}',(facebox[0],facebox[1]-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2,cv2.LINE_AA)
        cv2.imshow("Detect age and gender",resultImg)

        if cv2.waitKey(33) & 0xFF == ord('q'):
            break
        
        
cv2.destroyAllWindows()