import os
import kivy
import numpy as np
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.label import Label
import pandas as pd
import numpy as np
import keras as kr
import tensorflow as tf
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation,Flatten
from tensorflow.keras.layers import Conv2D,MaxPooling2D
from keras.models import load_model
import cv2
from kivy.app import App
from kivy.config import Config 
from kivy.uix.button import Button
from kivy.uix.floatlayout import FloatLayout
from skimage.transform import resize
from kivy.uix.filechooser import FileChooserListView, FileChooserIconView
import pathlib
from skimage import color

class MyApp(App):

    def importer(self,instance):
        self.camera = cv2.VideoCapture(0)
        return_value, image = self.camera.read()
        cv2.imwrite('C:/Users/ASUS/Desktop/Master/S3/El far/Project_finalle_Hamza_Kardate/App_mobile_Hamza_Kardate/opencv_image/opencv.png', image)
        del(self.camera)
    def press(self,instance):
        print("Pressed")
    def pred(self,instance):
        b="opencv_image/opencv.png"
        img=plt.imread(b)
        resized_image = resize(img,(32,32,3))
        imgGray = color.rgb2gray(resized_image)
        tflite_model='model2.tflite'
        interpreter = tf.lite.Interpreter(model_path=tflite_model)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        g=imgGray.reshape((1,32,32,1))
        to_predict = np.array(np.array(g),dtype='float32')
        interpreter.set_tensor(input_details[0]['index'], to_predict)
        interpreter.invoke()
        tflite_results = interpreter.get_tensor(output_details[0]['index'])
        
        a=np.argmax(tflite_results,axis=1)
        class_list=["Class 0","Class 1","Class 2","Class 3","Class 4"]
        print(class_list[int(a)])     
        self.title = 'Pred'
        self.box.add_widget(Label(text=f'{class_list[int(a)]}', pos_hint={'center_x':0.3, 'center_y':0.65}))
        return self.box      
    def select_image(self,instance):
        #module = load_model('tfmodel.lt')
        path =self.f.selection[0]
        img=plt.imread(path)
        resized_image = resize(img,(32,32,3))
        imgGray = color.rgb2gray(resized_image)
        tflite_model='model2.tflite'
        interpreter = tf.lite.Interpreter(model_path=tflite_model)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        g=imgGray.reshape((1,32,32,1))
        to_predict = np.array(np.array(g),dtype='float32')
        interpreter.set_tensor(input_details[0]['index'], to_predict)
        interpreter.invoke()
        tflite_results = interpreter.get_tensor(output_details[0]['index'])
        
        a=np.argmax(tflite_results,axis=1)

        #predictions=module.predict(np.array([resized_image]))

        class_list=["Class 0","Class 1","Class 2","Class 3","Class 4"]
        list=[0,1,2,3,4]
        X=tflite_results
        for i in range(5):
          for j in range(5):
            if X[0][list[i]] > X[0][list[j]]:
              temp=list[i]
              list[i]=list[j]
              list[j]=temp
        
        
        for i in range(5):
          self.box.add_widget(Label(text=class_list[list[i]], pos_hint={'center_x':0.2, 'center_y':1-(0.4+i*0.05)}))
          self.box.add_widget(Label(text=str(round(tflite_results[0][list[i]]*100 , 2)), pos_hint={'center_x':0.6, 'center_y':1-(0.4+i*0.05)}))
        #self.box.add_widget(Label(text=f'{class_list[0]}', pos_hint={'center_x':0.3, 'center_y':0.5}))
        return self.box
    def build(self):
        
        self.title = 'Free Positioning'
        self.box = FloatLayout(size=(300, 600))
        self.f =FileChooserListView( path = 'C:/Users/ASUS/Desktop/Master/S3/El far/Project_finalle_Hamza_Kardate/App_mobile_Hamza_Kardate/file_app')
        self.box.add_widget(self.f)
        butt1=Button(text='Image Webcam', size_hint=(0.5, 0.2), pos=(0, 0))
        butt1.bind(on_press=self.importer)
        self.box.add_widget(butt1)
        butt2=Button(text='Predect Image', size_hint=(0.5, 0.2), pos=(150, 0))
        butt2.bind(on_press=self.pred)
        self.box.add_widget(butt2)
        butt0=Button(text='Seclect and Predect Image', size_hint=(1, 0.2), pos=(0, 100))
        butt0.bind(on_press=self.select_image)
        self.box.add_widget(butt0)
        return self.box    
    Config.set('graphics', 'width', '300') 
    Config.set('graphics', 'height', '600')


if __name__ == '__main__':
    MyApp().run()

