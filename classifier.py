#general imports
from keras.utils import np_utils
from sklearn.datasets import load_files
import numpy as np
from glob import glob

#imports for dog and human detector functions
from keras.applications.resnet50 import preprocess_input as preprocess_input_resnet50
from keras.preprocessing import image as image_processor
from tqdm import tqdm
import cv2

#imports for loading the trained keras models:
from keras.models import load_model

#import inceptionv3 bottleneck features to run new images through inceptionv3, then run it through my models
from keras.applications.inception_v3 import preprocess_input as preprocess_input_inceptionv3

class classifier_network:

    def __init__(self):
        #dog detector model:
        self.ResNet50_model = load_model('resnet50_dog_identifier_model.h5')
        self.face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

        #inception model without fully connected layers
        self.inception_transferred = load_model('inceptionv3_dog_classifier_convolution_layers.h5')

        #dog names ordered for net:
        with open('dog_names.txt', 'r') as file:
            self.dog_names = []
            for line in file:
                self.dog_names.append(line)

        #my model pre-trained:
        self.inception_model = load_model('inception_model_classifier.h5')

    def path_to_tensor(self, img_path):
        # loads RGB image as PIL.Image.Image type
        img = image_processor.load_img(img_path, target_size=(224, 224))
        # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
        x = image_processor.img_to_array(img)
        # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
        return np.expand_dims(x, axis=0)

    def paths_to_tensor(self, img_paths):
        list_of_tensors = [self.path_to_tensor(img_path) for img_path in tqdm(img_paths)]
        return np.vstack(list_of_tensors)

    def ResNet50_predict_labels(self, img_path):
        # returns prediction vector for image located at img_path
        img = preprocess_input_resnet50(self.path_to_tensor(img_path))
        return np.argmax(self.ResNet50_model.predict(img))

    def dog_detector(self, img_path):
        prediction = self.ResNet50_predict_labels(img_path)
        return ((prediction <= 268) & (prediction >= 151))

    def face_detector(self, img_path):
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray)
        return len(faces) > 0

    def extract_InceptionV3(self, tensor):
        return self.inception_transferred.predict(preprocess_input_inceptionv3(tensor))

    def inception_predict_breed(self, image_path):
        botneck_feature = self.extract_InceptionV3(self.path_to_tensor(image_path))
        predicted_breed = np.argmax(self.inception_model.predict(botneck_feature))
        return self.dog_names[predicted_breed]
