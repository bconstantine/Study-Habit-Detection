#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:





# In[1]:


import os
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets                     # uic
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QWidget, QLabel, QVBoxLayout, QMessageBox)              # +++
import datetime
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import os
import mediapipe as mp #trial: for now we will try using mediapipe as pretrained model
import math as m
from win10toast import ToastNotifier
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard, EarlyStopping

#using holistic model to additionally detect face landmark rather than pose only
#if it turns out too heavy, then switch holistic to pose in the future
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils #for drawing the landmark to the screen (opencv)

def draw_points(img, holisticOut, mp_holistic):
    '''
    Draw the detected result to opencv bgr image
    
    image: A three channel BGR image represented as numpy ndarray.
    holisticOut: the detected result of the holistic model
    
    no return, since img.flags.writeable is assumed to be True (from the mp_predict() below)
    '''
    #draw all: face, pose (body), right and left hand
    
    #skip the face part, because we will draw from long range
    mp_drawing.draw_landmarks(img, holisticOut.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                             mp_drawing.DrawingSpec(color=(80,112,4), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(4,176,176), thickness=1, circle_radius=1)
                             ) # face'''
    mp_drawing.draw_landmarks(img, holisticOut.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(3,37,205), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(3,158,205), thickness=2, circle_radius=2)
                             ) # pose
    
    #left and right hand
    mp_drawing.draw_landmarks(img, holisticOut.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                              mp_drawing.DrawingSpec(color=(255,64,90), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(255,154,167), thickness=2, circle_radius=2)
                             ) 
    mp_drawing.draw_landmarks(img, holisticOut.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(255,123,21), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(255,192,144), thickness=2, circle_radius=2)
                             ) 
    '''
    if(holisticOut.pose_landmarks):
        h,w = img.shape[:2]
        lm = holisticOut.pose_landmarks
        lmPose = mp_holistic.PoseLandmark
        cv2.line(img, (int(lm.landmark[lmPose.LEFT_SHOULDER].x * w), int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)), (int(lm.landmark[lmPose.LEFT_EAR].x * w), int(lm.landmark[lmPose.LEFT_EAR].y * h)), (3,158,205), 2)
        #cv2.line(img, (int(lm.landmark[lmPose.LEFT_SHOULDER].x * w), int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)), (int(lm.landmark[lmPose.LEFT_SHOULDER].x * w), int(lm.landmark[lmPose.LEFT_SHOULDER].y * h) - 100), (3,158,205), 2)
        #cv2.line(img, (int(lm.landmark[lmPose.LEFT_HIP].x * w), int(lm.landmark[lmPose.LEFT_HIP].y * h)), (int(lm.landmark[lmPose.LEFT_SHOULDER].x * w), int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)), (3,158,205), 2)
        #cv2.line(img, (int(lm.landmark[lmPose.LEFT_HIP].x * w), int(lm.landmark[lmPose.LEFT_HIP].y * h)), (int(lm.landmark[lmPose.LEFT_HIP].x * w), int(lm.landmark[lmPose.LEFT_HIP].y * h) - 100), (3,158,205), 2)
    '''
    
def mp_predict(img, holistic):
    '''
    launched one cycle of holistic prediction in mediapipe
    
    image: A three channel BGR image represented as numpy ndarray.
    holistic: model 
    '''
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #mediapipe works in rgb, convert first
    
    #beware! Image in Opencv is passed by reference, any modification to the data will change it
    #use img.flags.writeable = False to turn off
    img.flags.writeable = False                  
    holisticOut = holistic.process(img)                   # Make prediction, requires rgb
    img.flags.writeable = True
    
    #convert back
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img, holisticOut #holisticOut will be passed to the draw_points

#helper functions
def findDistance(x1,y1,x2,y2):
    dist = m.sqrt((x2-x1)**2 + (y2-y1)**2)
    return dist
def findAngle(x1, y1, x2, y2):
    theta = m.acos( (y2 -y1)*(-y1) / (m.sqrt((x2 - x1)**2 + (y2 - y1)**2 ) * y1))
    #degree = int(180/m.pi)*theta
    #return degree
    return theta

holistic_params = {
    'min_detection_confidence' : 0.5, 
    'min_tracking_confidence': 0.5
}

def extract_point_data(holisticOut, mp_holistic, h, w):
    #will be in order of [pose, face, left_hand, right_hand]
    #face  468 * [x,y,z]
    #pose 33 * [x,y,z,visibility]
    #left hand right hand 21 [x,y,z]
    pose_data = np.array([[hOut.x, hOut.y, hOut.z, hOut.visibility] for hOut in holisticOut.pose_landmarks.landmark]).flatten() if holisticOut.pose_landmarks else np.zeros(33*4)
    face_data = np.array([[hOut.x, hOut.y, hOut.z] for hOut in holisticOut.face_landmarks.landmark]).flatten() if holisticOut.face_landmarks else np.zeros(468*3)
    lh_data = np.array([[hOut.x, hOut.y, hOut.z] for hOut in holisticOut.left_hand_landmarks.landmark]).flatten() if holisticOut.left_hand_landmarks else np.zeros(21*3)
    rh_data = np.array([[hOut.x, hOut.y, hOut.z] for hOut in holisticOut.right_hand_landmarks.landmark]).flatten() if holisticOut.right_hand_landmarks else np.zeros(21*3)
    lm = holisticOut.pose_landmarks
    lmPose = mp_holistic.PoseLandmark
    neck_angle_data = np.array([findAngle(int(lm.landmark[lmPose.LEFT_SHOULDER].x * w), int(lm.landmark[lmPose.LEFT_SHOULDER].y * h), int(lm.landmark[lmPose.LEFT_EAR].x * w), int(lm.landmark[lmPose.LEFT_EAR].y * h))]) if holisticOut.pose_landmarks else np.zeros(1)
    return np.concatenate([pose_data, face_data, lh_data, rh_data])
    #return np.concatenate([pose_data, lh_data, rh_data, neck_angle_data])

toast = ToastNotifier()

def sendWarning(action):
    toast.show_toast(f'Bad study posture: {action}', 'Your study posture is bad for some time, please fix it!', duration = 5, threaded = True)

#BodyPostureDetection
actions = np.array(['good', 'cheek', 'forehead'])
def buildModel(optimizer_params = 'Adam', loss_params = 'categorical_crossentropy', metrics_params = ['categorical_accuracy']):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))
    model.compile(optimizer=optimizer_params, loss=loss_params, metrics=metrics_params)
    print(model.summary())
    return model
colors = [(245,117,16), (117,245,16), (16,117,245), (100,100,100)]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame

threshold = 0.5 #for predict
frame_per_video = 30

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(640, 480) #525, 386
        self.horizontalLayout = QtWidgets.QHBoxLayout(Form)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")

        self.image_label = QtWidgets.QLabel(Form)
        self.image_label.setObjectName("image_label")
        self.verticalLayout.addWidget(self.image_label)

        self.control_bt = QtWidgets.QPushButton(Form)
        self.control_bt.setObjectName("control_bt")
        self.verticalLayout.addWidget(self.control_bt)

        self.capture = QtWidgets.QPushButton(Form)
        self.capture.setObjectName("capture")
        self.verticalLayout.addWidget(self.capture)

        self.horizontalLayout.addLayout(self.verticalLayout)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form",     "Cam view"))
        self.control_bt.setText(_translate("Form", "Start or Stop"))
        self.capture.setText(_translate("Form",    "Capture"))

class video (QtWidgets.QDialog, Ui_Form):
    def __init__(self):
        super().__init__()                  
        self.setupUi(self)                                     

        self.control_bt.clicked.connect(self.start_webcam)
        self.capture.clicked.connect(self.capture_image)
        self.capture.clicked.connect(self.startUIWindow)       

        self.image_label.setScaledContents(True)
        self.startCapture = False
        self.cap = None                                        
        
        self.timer = QtCore.QTimer(self, interval=5)
        self.timer.timeout.connect(self.update_frame)
        self._image_counter = 0
        
        #holistic part
        self.holistic = mp_holistic.Holistic(min_detection_confidence=holistic_params['min_detection_confidence'], min_tracking_confidence=holistic_params['min_tracking_confidence'])
        self.AImodel= buildModel()
        self.AImodel.load_weights('studyface_more.h5')
        self.sequence = []
        self.predictions = []
        self.prevPosture = 0
        self.postureRepetition = 0
    
    def closeEvent(self, event):
        close = QMessageBox()
        close.setText("Do you want to quit?")
        close.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
        close = close.exec()
        
        if close == QMessageBox.Yes:
            event.accept()
            if self.cap != None:
                self.cap.release()
        else:
            event.ignore()
    @QtCore.pyqtSlot()
    def start_webcam(self):
        if self.startCapture == False:
            if self.cap is None:
                self.cap = cv2.VideoCapture(0)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            self.startCapture = True
            self.timer.start()
        else:
            self.startCapture = False

    @QtCore.pyqtSlot()
    def update_frame(self):
        if self.startCapture: 
            ret, frame = self.cap.read()
            #simage     = cv2.flip(image, 1)
            img, holisticOut = mp_predict(frame, self.holistic)
            draw_points(img, holisticOut, mp_holistic)
            
            h,w = img.shape[:2]
            np_point_data = extract_point_data(holisticOut, mp_holistic, h, w)
            self.sequence.append(np_point_data)
            self.sequence = self.sequence[-frame_per_video:]
            
            if len(self.sequence) == frame_per_video:
                #print('reach here!')
                res = self.AImodel.predict(np.expand_dims(self.sequence, axis=0), verbose = 0)[0] #we need to use [0] because dimens
                #print(actions[np.argmax(res)])
                curPred = np.argmax(res)
                self.predictions.append(curPred)


                #3. Viz logic
                #if np.unique(self.predictions[-10:])[0]==np.argmax(curPred): 
                if res[np.argmax(res)] > threshold: 
                    if self.prevPosture == curPred:
                        self.postureRepetition += 1
                        if(self.prevPosture in [1,2] and self.postureRepetition == 100):
                            sendWarning(actions[curPred])
                    else:
                        self.prevPosture = curPred
                        self.postureRepetition = 0

                    
                # Viz probabilities
                img = prob_viz(res, actions, img, colors)
                cv2.putText(img, f'Current detected action: {actions[curPred]}, repetition: {self.postureRepetition}', (15,12), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            self.displayImage(img, True)
        else:
            img = np.zeros((480, 640, 3))
            self.displayImage(img, True)

    @QtCore.pyqtSlot()
    def capture_image(self):
        flag, frame = self.cap.read()
        path = r'screenshots' 
        if flag:
            QtWidgets.QApplication.beep()
            img, holisticOut = mp_predict(frame, self.holistic)
            draw_points(img, holisticOut, mp_holistic)
            
            h,w = img.shape[:2]
            np_point_data = extract_point_data(holisticOut, mp_holistic, h, w)
            self.sequence.append(np_point_data)
            self.sequence = self.sequence[-frame_per_video:]
            
            if len(self.sequence) == frame_per_video:
                #print('reach here!')
                res = self.AImodel.predict(np.expand_dims(self.sequence, axis=0), verbose = 0)[0] #we need to use [0] because dimens
                #print(actions[np.argmax(res)])
                self.predictions.append(np.argmax(res))


                #3. Viz logic
                '''
                if np.unique(self.predictions[-10:])[0]==np.argmax(res): 
                    if res[np.argmax(res)] > threshold: 
                        if len(sentence) > 0: 
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                        else:
                            sentence.append(actions[np.argmax(res)])

                if len(sentence) > 5: 
                    sentence = sentence[-5:]
                '''
                # Viz probabilities
                img = prob_viz(res, actions, img, colors)

            x = str(datetime.datetime.now())
            a = ""
            for i in range(len(x)):
                if x[i] in ['-',' ',':','.']:
                    a+="_"
                else:
                    a+=x[i]

            name = a +".jpg"
            print(frame)
            print(cv2.imwrite(os.path.join(path, name), img))
            self._image_counter += 1

    def displayImage(self, img, window=True):
        qformat = QtGui.QImage.Format_Indexed8
        if not (img is None):
            if len(img.shape)==3 :
                if img.shape[2]==4:
                    qformat = QtGui.QImage.Format_RGBA8888
                else:
                    qformat = QtGui.QImage.Format_RGB888
            outImage = QtGui.QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
            #print('dim1:'+str(img.shape[1]))
            #print('dim0:'+str(img.shape[0]))
            outImage = outImage.rgbSwapped()
            if window:
                self.image_label.setPixmap(QtGui.QPixmap.fromImage(outImage))

    def startUIWindow(self):
        self.Window = UIWindow()                               # - self
        self.setWindowTitle("UIWindow")
        self.Window.ToolsBTN.clicked.connect(self.goWindow1)
        self.hide()
        self.Window.show()
    def goWindow1(self):
        self.show()
        self.Window.hide()


class UIWindow(QWidget):
    def __init__(self, parent=None):
        super(UIWindow, self).__init__(parent)

        self.resize(300, 300)
        self.label = QLabel("Screen captured!", alignment=QtCore.Qt.AlignCenter)

        self.ToolsBTN = QPushButton('Back to main window')
#        self.ToolsBTN.move(50, 350)

        self.v_box = QVBoxLayout()
        self.v_box.addWidget(self.label)
        self.v_box.addWidget(self.ToolsBTN)
        self.setLayout(self.v_box)


if __name__=='__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = video()
    window.setWindowTitle('Study Habit Detection')
    window.show()
    sys.exit(app.exec_())


# In[2]:





# In[ ]:




