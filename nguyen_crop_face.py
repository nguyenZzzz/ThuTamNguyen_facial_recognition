import cv2
import os


classifier = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")

dirFace = './cropped_face'

# Create if there is no cropped face directory
if not os.path.exists(dirFace):
    os.mkdir(dirFace)
    print("Directory " , dirFace ,  " Created ")
else:    
    print("Directory " , dirFace ,  " has found.")

webcam = cv2.VideoCapture(0) # Camera 0 according to USB port
# video = cv2.VideoCapture(r"use full windows path") # video path

while (True):
    (f, im) = webcam.read() # f returns only True, False according to video access
    # (f, im) = video.read() # video 

    if f != True:
        break

    # im=cv2.flip(im,1,0) #if you would like to give mirror effect

    # detectfaces 
    faces = classifier.detectMultiScale(
        im, # stream 
        scaleFactor=1.10, # change these parameters to improve your video processing performance
        minNeighbors=20, 
        minSize=(30, 30) # min image detection size
        ) 

    # Draw rectangles around each face
    for (x, y, w, h) in faces:

        cv2.rectangle(im, (x, y), (x + w, y + h),(255,255,255),thickness=1)
        # saving faces according to detected coordinates 
        sub_face = im[y:y+h, x:x+w]
        FaceFileName = "cropped_face/face_" + str(y+x) + ".jpg" # folder path and random name image
        cv2.imwrite(FaceFileName, sub_face)

    # Video Window
    cv2.imshow('Video Stream',im)
    key = cv2.waitKey(1) & 0xFF
    # q for exit
    if key == ord('q'): 
        break

webcam.release()