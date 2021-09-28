import streamlit as st
from streamlit.proto.Button_pb2 import Button
import numpy as np
import cv2
import face_recognition as fr
import datetime
import glob, os
from PIL import Image, ImageDraw, ImageColor

########################################################################################

def make_up(image, eye_brow_fill, lips_fill):
    # Find all facial features in all the faces in the image
    face_landmarks_list = fr.face_landmarks(image)

    pil_image = Image.fromarray(image)
    for face_landmarks in face_landmarks_list:
        d = ImageDraw.Draw(pil_image, 'RGBA')

        # Make the eyebrows into a nightmare
        d.polygon(face_landmarks['left_eyebrow'], fill=eye_brow_fill) #(RGBA) 
        d.polygon(face_landmarks['right_eyebrow'], fill=eye_brow_fill)

        # Gloss the lips
        d.polygon(face_landmarks['top_lip'], fill=lips_fill)
        d.polygon(face_landmarks['bottom_lip'], fill=lips_fill)

        image = np.array(pil_image)
    return image


def paste_picture_in_picture(front_path, frame, scale, postition_in_background = (0,0)):
    from PIL import Image

    frontImage = Image.open(front_path)

    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    background = Image.fromarray(frame)
    
    # Scale:
    frontImage = frontImage.resize((int(frontImage.width*scale), int(frontImage.height*scale)))

    # (x1, y1):  position on bacground image
    (x1, y1) = postition_in_background
    x = int(x1 - frontImage.width/2)
    y = int(y1 - frontImage.height/2)
    
    # Paste the frontImage at (width, height)
    background.paste(frontImage, (x, y), frontImage)

    # Convert image to array 
    frame = np.array(background)
    return frame

def centroid(points_list):
     _x_list = [vertex [0] for vertex in points_list]
     _y_list = [vertex [1] for vertex in points_list]
     _len = len(points_list)
     _x = sum(_x_list) / _len
     _y = sum(_y_list) / _len
     return(_x, _y)

########################################################################################

MENU = ['Tam']
choice = st.sidebar.selectbox ('Please choose: ', MENU)
icon_path = './streamlit/scanning face_2.gif'

if choice=='Tam':
    col1, col2 = st.columns((2,1))
    with col1:
        st.title("FACE RECOGNITION +")
        st.subheader("Nguyen - Thu - Tam")
    
    with col2:
        st.image(icon_path)

    ############ SELECT MODE ############ 
    # st.subheader('You want to play with???', )
    mode_list = ['Face Recognition', 'Face Make-up']
    mode = st.selectbox('You want to play with???', mode_list)    

    if mode == 'Face Recognition':
        ############ FACE RECOGNITION ############ 

        ##### MAKE TRAINING DATA #####
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


        ##### WEBCAM #####
        show = st.checkbox('Show!')
        st.text('Press: "Space" to capture')

        FRAME_WINDOW = st.image([])
        cap = cv2.VideoCapture(1) # device 1/2
        while show:
            ret, frame = cap.read()

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
            text = f'Hello {len(face_locations)} human!'
            cv2.putText(frame, 
                            text, (0, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0,255,0), 2)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            FRAME_WINDOW.image(frame)
            # cv2.imshow('Input', frame)
        else:
            cap.release()
            cv2.destroyAllWindows()

        cap.release()
        cv2.destroyAllWindows()

    elif mode == 'Face Make-up':
        ############ FACE MAKE-UP ############ 
        makeup = st.checkbox('Make a nightmare face!')
        if makeup:
            col11, col22 = st.columns(2)
            ##### EYE-BROW SETUP #####
            with col11:
                # Use eye brow or not:

                # eye_brow color
                eye_brow_hex_cl = st.color_picker('EYEBROWS', key=1)
                eye_brow_rgb_cl = ImageColor.getcolor(eye_brow_hex_cl, 'RGB')

                # lighter --> darker:
                eye_brow_alpha = st.slider('Lighter --> Darker:',0,255, key=1, value=150)

                eye_brow_fill = eye_brow_rgb_cl+(eye_brow_alpha,)

            ##### LIPS SETUP #####
            with col22:
                # lips color
                lips_hex_cl = st.color_picker('LIPS', key=2)
                lips_rgb_cl = ImageColor.getcolor(lips_hex_cl, 'RGB')

                # lighter --> darker:
                lips_alpha = st.slider('Lighter --> Darker:',0,255, key=2, value=150)

                lips_fill = lips_rgb_cl+(lips_alpha,)

        ############ FACE MASK SETUP ############ 
        mask = st.checkbox('Covid 19 mode!')

        if mask:
            mask_scale = st.slider('',0.05,3.2,value=0.45,step = 0.01, key=3)

        ##### WEBCAM #####
        show = st.checkbox('Show webcam!')
        st.text('Press: "Space" to capture')

        FRAME_WINDOW = st.image([])
        cap = cv2.VideoCapture(1) # device 1/2
        while show:
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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

            ##### FACE MAKE-UP #####
            if makeup:
                frame = make_up(frame, eye_brow_fill, lips_fill)
            
            ##### FACE MASK #####
            if mask:
                # GET LOCATION OF EYES, NOSE, LIPS,...
                try:
                    face_landmarks_list = fr.face_landmarks(frame)
                    top_lip_locations = face_landmarks_list[0]['top_lip']

                    # Get center of top_lip
                    center_point = centroid(top_lip_locations)

                    # mark for face:
                    mask_path = '/home/tamtran/Documents/GitHub/ThuTamNguyen_facial_recognition/mask/mask.png'
                    frame = paste_picture_in_picture(mask_path, frame, mask_scale, center_point)
                    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                except:
                    pass
    
            FRAME_WINDOW.image(frame)

        else:
            cap.release()
            cv2.destroyAllWindows()

        cap.release()
        cv2.destroyAllWindows()



