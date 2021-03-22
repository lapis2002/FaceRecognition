# Face Recognition
Face Recognition using DFSD and FaceNet.
#####**Accuracy: > 90%**
## Dependencies

- Jupyter
- Tensorflow
- Keras
- Matplotlib
- Face Detection (DFSD - PyTorch)
- NumPy
- OpenCV

## Face Detection
I have tried using MTCNN and Cascade Classifier of OpenCV for face detection but both could not detect side face effectively.

While search for other models, I found a face detection network developed by Tencent called [DSFD: Dual Shot Face Detector](https://github.com/Tencent/FaceDetection-DSFD) (2018).
More details about the DSFD can be viewed at [paper DSFD: Dual Shot Face Detector](https://arxiv.org/abs/1810.10220)

An example of side face that DSFD can detect but MTCNN and Cascade Classifier cannot:

![Side face detection](./testedFace.png?raw=true "Side face detection")

DSFD can be easily installed with pip:
`!pip install git+https://github.com/hukkelas/DSFD-Pytorch-Inference.git`

Then:
`import face_detection`

## Face Recognition

FaceNet is a face recognition system developed in 2015 by Google researchers that achieved then state-of-the-art results on a range of face recognition benchmark datasets.

Paper about FaceNet can be accessed at ["FaceNet: A Unified Embedding for Face Recognition and Clustering."](https://arxiv.org/abs/1503.03832)

I used pre-trained [Keras FaceNet by Hiroki Taniai](https://github.com/nyoki-mtl/keras-facenet). His project provides a script for converting the Inception ResNet v1 model to Keras. I downloaded the Keras model from [here](https://drive.google.com/drive/folders/1pwQ3H4aJ8a6yyJHZkTwtjcL4wYWQb7bn).

The expected input images is color, has their pixel values whitened (standardized across all three channels), and to have a square shape of 160×160 pixels.

After using FaceNet model to create a face embedding for each detected face, I used a Linear Support Vector Machine (SVM) classifier model to predict the identity of a given face.