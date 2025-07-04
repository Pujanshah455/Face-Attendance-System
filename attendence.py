import cv2
import face_recognition
import os
from datetime import datetime
import numpy as np

images=[]
classNames=[]

path = '_23AttendenseSystem\imagedata'
myList = os.listdir('_23AttendenseSystem\imagedata')

for person in myList:
    # img = cv2.imread(f"{path}/{person}")
    img = face_recognition.load_image_file(f"{path}/{person}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (600, 500))
    images.append(img)
    classNames.append(os.path.splitext(person)[0])
    

def findEncoding(images):
    encodeList = []
    for img in images:
        imgencode = face_recognition.face_encodings(img)
        if len(imgencode) != 0:
            # if in any cases given data photo is blur or not capable to recognize than list become empty so, we need to use this
            encodeList.append(imgencode[0])
    
    return encodeList


def markAttendence(name):
    with open('_23AttendenseSystem\Attendence.txt', 'r+') as f:
        data = f.readlines()
        nameList = []
        for line in data:
            nameInFile = line.split(',')[0]
            nameList.append(nameInFile)
        if name not in nameList:
            nameList.append(name)
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')
    
encodeList = findEncoding(images)

cap = cv2.VideoCapture(0)

while True:
    res, frame = cap.read()
    
    frame_small = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # frame = cv2.resize(frame, (600, 500)) if we do this than video become laggy
    frame_small = cv2.resize(frame_small, (0, 0), fx=0.25, fy=0.25)
    
    frameLoc = face_recognition.face_locations(frame_small) # (top, right, bottom, left)
    frameEncode = face_recognition.face_encodings(frame_small, frameLoc) 
    # if we give framLoc than encoder not to find face direct it encode. So, our code become more faster
    
    for faceLoc, faceEncode in zip(frameLoc, frameEncode):
        # we done this because in camera frame more than one faces than we get for perticular face that location and encode in list  
        matches = face_recognition.compare_faces(encodeList, faceEncode)
        faceDis = face_recognition.face_distance(encodeList, faceEncode)
        
        matchIdx = np.argmin(faceDis)
        if matches[matchIdx]:
            name = classNames[matchIdx].upper()
        else:
            name = 'unkonwn'
        
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4 # because first we multiply all pixel by(1/4=0.25)
        cv2.rectangle(frame, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=2)
        cv2.rectangle(frame, pt1=(x1, y2+35), pt2=(x2, y2), color=(0, 255, 0), thickness=-1)
        cv2.putText(frame, name, org=(x1+6, y2+20), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.6, color=(255, 255, 255), thickness=2)
        markAttendence(name)
        
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF==ord('x'):
        break
    
    

