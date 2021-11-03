import cv2 as cv
import numpy as np
import time

capture=cv.VideoCapture(0)
haar_cascade=cv.CascadeClassifier('haar_face.xml')

prev_frame_time = 0
new_frame_time = 0 

while True:
    isTrue, frame= capture.read()
    # cv.imshow('Live' , frame)
    gray=cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # cv.imshow('Gray Live', gray)
    canny=cv.Canny(frame, 125, 175)
    # cv.imshow('Canny Live', canny)

    #for fps
    new_frame_time= time.time()

    fps=1/(new_frame_time-prev_frame_time)

    prev_frame_time=new_frame_time

    fps=int (fps)
    fps=str (fps)


    face_rectangle=haar_cascade.detectMultiScale(gray, scaleFactor=1.1,minNeighbors=9)

    print(f'Number of faces found = {len(face_rectangle)}')


    for (x,y,w,h) in face_rectangle:
        # cv.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), thickness=2)
        x1=(2*x+w)/2
        y1=(2*y+h)/2
        r=int(w/2)
        cv.circle(frame, (int(x1),int(y1)), r, (255,0,0), thickness=2)

        # x2=(2*x+w)/2
        # y2=(y+h)+25




    cv.putText(frame, f'People:= {len(face_rectangle)} and fps:= {fps}', (50,50), cv.FONT_HERSHEY_TRIPLEX, 0.5, (100,255,130), 1)

    cv.imshow('Detected Live Video', frame)


    if cv.waitKey(20) & 0xFF==ord('d'):
        break

capture.release()
cv.destroyAllWindows()