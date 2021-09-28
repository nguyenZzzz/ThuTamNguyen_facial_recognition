import face_recognition as fr
import cv2
import datetime
import glob, os

################################## TRAIN FACE ##################################

known_face_encodings = []
known_face_names = []

data_dir = '/home/tamtran/Documents/GitHub/ThuTamNguyen_facial_recognition/Data'
class_names = os.listdir(data_dir)
print(class_names)

for class_name in class_names:
    class_dir = os.path.join(data_dir, class_name)
    all_file_path = glob.glob(class_dir+'/*.*')
    print(f'{class_name} has: {len(all_file_path)} images')
    if len(all_file_path) > 0:
        for image_path in all_file_path:
            image = fr.load_image_file(image_path)
            try:
                face_encoding = fr.face_encodings(image)[0]
                known_face_encodings.append(face_encoding)
                known_face_names.append(class_name)
            except:
                pass

print(f'len of known_face_encoding: {len(known_face_encodings)}')
print(f'len of known_face_names: {len(known_face_names)}')


################################## WEBCAM ##################################
cap = cv2.VideoCapture(1)
count = 0

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()

    # frame is now the image capture by the webcam (one frame of the video)

    c = cv2.waitKey(1)
		# Break when pressing ESC
    if c == 27:
        break

    # Take picture:
    if c == 32:
        time_now = str(datetime.datetime.now())
        img_name = time_now+'.png'
        cv2.imwrite(img_name, frame)    
        count = count+1
        print(f'image number: {count}')

    # FACE IDENTIFY:
    # Get face location and encode them
    face_locations = fr.face_locations(frame)
    face_encodings = fr.face_encodings(frame, face_locations)

    for (t,r,b,l), face_encoding in zip(face_locations, face_encodings):
        # Verify face and name:
        matches = fr.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
        name = 'Ai vay?'
        if True in matches:
            face_idx = matches.index(True)
            name = known_face_names[face_idx]
        
        # Draw rectangle:
        cv2.rectangle(frame, (l,t), (r, b), (0,0,0), 2)

        # Put name:
        cv2.putText(frame, 
					name, (l, t), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,255,0), 2)
    
    # Show number of people:
    text = f'Hello {len(face_locations)} human-being!'
    cv2.putText(frame, 
					text, (0, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,255,0), 2)
    
    cv2.imshow('Input', frame)

cap.release()
cv2.destroyAllWindows()
