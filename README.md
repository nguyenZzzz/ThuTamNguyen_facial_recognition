# FACIAL RECOGNITION

## Using face_recognotion package

### Package summary:
- Recognize and manipulate faces from Python with the world's simplest face recognition library
- Built using [dlib](http://dlib.net/)'s state-of-the-art face recognition built with deep learning. The model has an accuracy of 99.38% on the [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/) benchmark.
- Go to this [link](https://github.com/ageitgey/face_recognition) for more information
### Some main features:
- Find faces in picture:
  
  ![c32c74e54563859b2514e94d555ee3a0.png](https://cloud.githubusercontent.com/assets/896692/23625227/42c65360-025d-11e7-94ea-b12f28cb34b4.png)
  
- Find and manipulate facial features in pictture:
	Get locations and outlines of each person's eyes, nose, mouth and chin.
  
   ![gdfg](https://cloud.githubusercontent.com/assets/896692/23625282/7f2d79dc-025d-11e7-8728-d8924596f8fa.png)
- Connect with OpenCV for real-time face recognition:

	![](https://cloud.githubusercontent.com/assets/896692/24430398/36f0e3f0-13cb-11e7-8258-4d0c9ce1e419.gif)

### Workflow:

![image](https://user-images.githubusercontent.com/87942072/134812407-fc67b9a4-0bf8-4b5d-90fc-c8ec3bcd9c72.png)

-------------------------------------------------
## Image Detection with YOLOv3 & Classfication by DNN

### Pre-trained models
| Model name      | Training dataset |
|-----------------|------------------|
| Yolov3| COCO |
| Xception| ImageNet|

### Description
This model uses Xception architecture for Transfer Learning, and is trained on 300 images of 3 classes using the last 27 layers in the network. The classifier is a Dense layer with activaiton = 'softmax'

### Training Data
300 images of 3 classes have been used for the purpose of training the classifier of this project. The image is taken with either a laptop's webcam or cell-phone and cropped so that only the faces are in the images

### Performance
The accuracy for the final model is 100% 
