import os
import cv2
import numpy as np
import pandas as pd
import pytesseract
from fpdf import FPDF
from PyQt5 import QtCore, QtGui, QtWidgets
from tkinter import *
from tkinter import ttk
from tkinter.filedialog import asksaveasfile
from phones import *
from docx import Document
import re
from math import log, sqrt

#from PyQt5.QtWidgets import QApplication, QWidget, QLabel
#from PyQt5.QtGui import QIcon, QPixmap
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


class Ui_MainWindow(QtWidgets.QMainWindow):
    #__slots__ = "_texts"
    #def __init__(self):
        #self._texts = ""
        
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("ML App")
        MainWindow.setFixedSize(800, 600)
        MainWindow.setWindowFlags(QtCore.Qt.WindowCloseButtonHint | QtCore.Qt.WindowMinimizeButtonHint)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(210, 50, 801, 121))
        self.label.setStyleSheet("font: 20pt \"MS Shell Dlg 2\";\n"
                                 "color: rgb(170, 0, 0);")
        self.label.setObjectName("label")

        self.imageLabel = QtWidgets.QLabel(self.centralwidget)
        self.imageLabel.setGeometry(QtCore.QRect(70,150,800,300))
        self.mainImage = QtGui.QPixmap('Image.png')
        self.imageLabel.setPixmap(self.mainImage)
        
        
        ################## FACE DETECTION #####################################
        
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(0, 0, 800, 600))
        self.frame.setStyleSheet("background-color: rgb(240, 240, 240);")
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.pushButton = QtWidgets.QPushButton(self.frame)
        self.pushButton.setGeometry(QtCore.QRect(111, 50, 151, 51))
        self.pushButton.setStyleSheet("color: rgb(0, 0, 0);\n"
"font: 14pt \"MS Shell Dlg 2\";")
        self.pushButton.setObjectName("pushButton")
        self.label_2 = QtWidgets.QLabel(self.frame)
        self.label_2.setGeometry(QtCore.QRect(13, 132, 350, 301))
        self.label_2.setStyleSheet("color: rgb(0, 0, 0);")
        self.label_2.setObjectName("label_2")
        self.label_2.setStyleSheet("border: 2px solid black")
        self.label_3 = QtWidgets.QLabel(self.frame)
        self.label_3.setGeometry(QtCore.QRect(436, 132, 350, 301))
        self.label_3.setStyleSheet("color: rgb(0, 0, 0);")
        self.label_3.setObjectName("label_3")
        self.label_3.setStyleSheet("border: 2px solid black")
        self.pushButton_2 = QtWidgets.QPushButton(self.frame)
        self.pushButton_2.setGeometry(QtCore.QRect(535, 50, 151, 51))
        self.pushButton_2.setStyleSheet("color: rgb(0, 0, 0);\n"
"font: 14pt \"MS Shell Dlg 2\";")
        self.pushButton_2.setObjectName("pushButton_2")

        #######################################################################
        ######################## FACE RECOGNITION #############################

        self.frame_2 = QtWidgets.QFrame(self.frame)
        self.frame_2.setGeometry(QtCore.QRect(0, 0, 800, 600))
        self.frame_2.setStyleSheet("background-color: rgb(240, 240, 240);")
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.frame_2)
        self.pushButton_3.setGeometry(QtCore.QRect(120, 150, 200, 50))
        self.pushButton_3.setStyleSheet("color: rgb(0, 0, 0);\n"
"font: 14pt \"MS Shell Dlg 2\";")
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(self.frame_2)
        self.pushButton_4.setGeometry(QtCore.QRect(490, 150, 200, 50))
        self.pushButton_4.setStyleSheet("color: rgb(0, 0, 0);\n"
"font: 14pt \"MS Shell Dlg 2\";")
        self.pushButton_4.setObjectName("pushButton_4")
        self.textEdit_19 = QtWidgets.QTextEdit(self.frame_2)
        self.textEdit_19.setGeometry(QtCore.QRect(300, 80, 210, 50))
        self.textEdit_19.setStyleSheet("color: rgb(0, 0, 0); font: 30px \"MS Shell Dlg 2\";")
        self.textEdit_19.setObjectName("textEdit_2")

        #######################################################################
        ########################### TEXT CLASSIFICATION #######################

        self.frame_3 = QtWidgets.QFrame(self.frame_2)       
        self.frame_3.setGeometry(QtCore.QRect(0, 0, 800, 600))
        self.frame_3.setStyleSheet("background-color: rgb(240, 240, 240);")
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.pushButton_5 = QtWidgets.QPushButton(self.frame_3)
        self.pushButton_5.setGeometry(QtCore.QRect(342, 220, 120, 40))
        self.pushButton_5.setStyleSheet("color: rgb(0, 0, 0);\n"
"font: 14pt \"MS Shell Dlg 2\";")
        self.pushButton_5.setObjectName("pushButton_5")
        self.textEdit = QtWidgets.QTextEdit(self.frame_3)
        self.textEdit.setGeometry(QtCore.QRect(500, 150, 261, 201))
        self.textEdit.setStyleSheet("color: rgb(0, 0, 0) font: 10pt \"MS Shell Dlg 2\";\n"
"font: 14pt \"MS Shell Dlg 2\";")
        self.textEdit.setObjectName("textEdit")
        self.textEdit_2 = QtWidgets.QTextEdit(self.frame_3)
        self.textEdit_2.setGeometry(QtCore.QRect(39, 150, 261, 201))
        self.textEdit_2.setStyleSheet("color: rgb(0, 0, 0) font: 10pt \"MS Shell Dlg 2\";")
        self.textEdit_2.setObjectName("textEdit_2")
        self.label_5 = QtWidgets.QLabel(self.frame_3)
        self.label_5.setGeometry(QtCore.QRect(133, 75, 101, 71))
        self.label_5.setStyleSheet("color: rgb(0, 0, 0);\n"
                                   "font: 20pt \"MS Shell Dlg 2\";")
        self.label_5.setObjectName("label_5")
        
        self.label_10 = QtWidgets.QLabel(self.frame_3)
        self.label_10.setGeometry(QtCore.QRect(590, 75, 101, 71))
        self.label_10.setStyleSheet("color: rgb(0, 0, 0);\n"
                                   "font: 20pt \"MS Shell Dlg 2\";")
        self.label_10.setObjectName("label_10")
        #self.label_5.setStyleSheet("border: 2px solid red")
        
        #######################################################################
        ###################### BANKER'S DATABASE ################################

        self.frame_4 = QtWidgets.QFrame(self.frame)
        self.frame_4.setGeometry(QtCore.QRect(0, 0, 800, 600))
        self.frame_4.setStyleSheet("background-color: rgb(240, 240, 240);")
        self.frame_4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_4.setObjectName("frame_4")
        self.pushButton_6 = QtWidgets.QPushButton(self.frame_4)
        self.pushButton_6.setGeometry(QtCore.QRect(220, 110, 381, 111))
        self.pushButton_6.setStyleSheet("color: rgb(0, 0, 0);\n"
                                        "font: 18pt \"MS Shell Dlg 2\";")
        self.pushButton_6.setObjectName("pushButton_6")
        
       
     
        #######################################################################
        ###################### IMAGE TO TEXT ##################################
        
        self.frame_5 = QtWidgets.QFrame(self.frame_4)       
        self.frame_5.setGeometry(QtCore.QRect(0, 0, 800, 600))
        self.frame_5.setStyleSheet("background-color: rgb(240, 240, 240);")
        self.frame_5.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_5.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_5.setObjectName("frame_5")
        self.pushButton_7 = QtWidgets.QPushButton(self.frame_5)
        self.pushButton_7.setGeometry(QtCore.QRect(111, 50, 151, 51))
        self.pushButton_7.setStyleSheet("color: rgb(0, 0, 0);\n"
                                        "font: 14pt \"MS Shell Dlg 2\";")
        self.pushButton_7.setObjectName("pushButton_7")
        self.label_7 = QtWidgets.QLabel(self.frame_5)
        self.label_7.setGeometry(QtCore.QRect(13, 132, 350, 301))
        self.label_7.setObjectName("label_7")
        self.label_7.setStyleSheet("border: 2px solid black")
        self.pushButton_8 = QtWidgets.QPushButton(self.frame_5)
        self.pushButton_8.setGeometry(QtCore.QRect(535, 50, 151, 51))
        self.pushButton_8.setStyleSheet("color: rgb(0, 0, 0);\n"
                                        "font: 14pt \"MS Shell Dlg 2\";")
        self.pushButton_8.setObjectName("pushButton_8")
        self.textEdit_9 = QtWidgets.QTextEdit(self.frame_5)
        self.textEdit_9.setGeometry(QtCore.QRect(436, 132, 350, 301))
        self.textEdit_9.setObjectName("label_8")
        self.textEdit_9.setStyleSheet("border: 2px solid black")
        self.pushButton_10 = QtWidgets.QPushButton(self.frame_5)
        self.pushButton_10.setGeometry(QtCore.QRect(410, 450, 151, 51))
        self.pushButton_10.setStyleSheet("color: rgb(0, 0, 0);\n"
                                        "font: 14pt \"MS Shell Dlg 2\";")
        self.pushButton_10.setObjectName("pushButton_10")
        self.pushButton_11 = QtWidgets.QPushButton(self.frame_5)
        self.pushButton_11.setGeometry(QtCore.QRect(600, 450, 151, 51))
        self.pushButton_11.setStyleSheet("color: rgb(0, 0, 0);\n"
                                        "font: 14pt \"MS Shell Dlg 2\";")
        self.pushButton_11.setObjectName("pushButton_11")
        
        #######################################################################
        ######################## TEXT SUMMARIZATION ###########################
        
        self.frame_6 = QtWidgets.QFrame(self.frame)
        self.frame_6.setGeometry(QtCore.QRect(0, 0, 800, 600))
        self.frame_6.setStyleSheet("background-color: rgb(240, 240, 240);")
        self.frame_6.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_6.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_6.setObjectName("frame_6")
        self.textEdit_3 = QtWidgets.QTextEdit(self.frame_6)
        self.textEdit_3.setGeometry(QtCore.QRect(20, 20, 760, 210))
        self.textEdit_3.setStyleSheet("color: rgb(0, 0, 0);")
        self.textEdit_3.setObjectName("textEdit_3")
        self.pushButton_9 = QtWidgets.QPushButton(self.frame_6)
        self.pushButton_9.setGeometry(QtCore.QRect(242, 265, 321, 40))
        self.pushButton_9.setStyleSheet("color: rgb(0, 0, 0);\n"
"font: 14pt \"MS Shell Dlg 2\";")
        self.pushButton_9.setObjectName("pushButton_9")
        self.textEdit_4 = QtWidgets.QTextEdit(self.frame_6)
        self.textEdit_4.setGeometry(QtCore.QRect(20, 340, 760, 210))
        self.textEdit_4.setStyleSheet("color: rgb(0, 0, 0);")
        self.textEdit_4.setObjectName("textEdit_4")

        #######################################################################

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 808, 21))
        self.menubar.setObjectName("menubar")
        self.menuImage_processing = QtWidgets.QMenu(self.menubar)
        self.menuImage_processing.setObjectName("menuImage_processing")
        self.menuText_classfication = QtWidgets.QMenu(self.menubar)
        self.menuText_classfication.setObjectName("menuText_classfication")
        self.menuOptions = QtWidgets.QMenu(self.menubar)
        self.menuOptions.setObjectName("menuOptions")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionFace_detect = QtWidgets.QAction(MainWindow)
        self.actionFace_detect.setObjectName("actionFace_detect")
        self.actionFace_recognition = QtWidgets.QAction(MainWindow)
        self.actionFace_recognition.setObjectName("actionFace_recognition")
        self.actionDigits_recognition = QtWidgets.QAction(MainWindow)
        self.actionDigits_recognition.setObjectName("actionDigits_recognition")
        self.actionOther_objects_detection = QtWidgets.QAction(MainWindow)
        self.actionOther_objects_detection.setObjectName("actionOther_objects_detection")
        self.actionHome = QtWidgets.QAction(MainWindow)
        self.actionHome.setObjectName("actionHome")
        self.actionExit = QtWidgets.QAction(MainWindow)
        self.actionExit.setObjectName("actionExit")
        self.actionText_summarizer = QtWidgets.QAction(MainWindow)
        self.actionText_summarizer.setObjectName("actionText_summarizer")
        self.actionText_classification = QtWidgets.QAction(MainWindow)
        self.actionText_classification.setObjectName("actionText_classification")
        self.menuImage_processing.addAction(self.actionFace_detect)
        self.menuImage_processing.addAction(self.actionFace_recognition)
        self.menuImage_processing.addAction(self.actionDigits_recognition)
        self.menuImage_processing.addAction(self.actionOther_objects_detection)
        self.menuText_classfication.addAction(self.actionText_summarizer)
        self.menuText_classfication.addAction(self.actionText_classification)
        self.menuOptions.addAction(self.actionHome)
        self.menuOptions.addAction(self.actionExit)
        self.menubar.addAction(self.menuOptions.menuAction())
        self.menubar.addAction(self.menuImage_processing.menuAction())
        self.menubar.addAction(self.menuText_classfication.menuAction())
        self.frame.hide()
        self.frame_2.hide()
        self.frame_3.hide()
        self.frame_4.hide()
        self.frame_5.hide()
        self.frame_6.hide()

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "Welcome to Banker's Assistant"))
        self.pushButton.setText(_translate("MainWindow", "Choose Image"))
        self.label_2.setText(_translate("MainWindow", ""))
        self.label_3.setText(_translate("MainWindow", ""))
        self.pushButton_2.setText(_translate("MainWindow", "Detect Faces"))

        self.pushButton_3.setText(_translate("MainWindow", "Face store"))
        self.pushButton_4.setText(_translate("MainWindow", "Face recognize"))

        self.pushButton_5.setText(_translate("MainWindow", "Classify"))

        self.label_5.setText(_translate("MainWindow", "Input"))
        self.label_10.setText(_translate("MainWindow", "Output"))

        self.pushButton_6.setText(_translate("MainWindow", "Launch Databse"))
        self.pushButton_7.setText(_translate("MainWindow", "Choose image"))
        self.label_7.setText(_translate("MainWindow", ""))
        self.pushButton_8.setText(_translate("MainWindow", "Image to text"))
        self.pushButton_10.setText(_translate("MainWindow", "Save as PDF"))
        self.pushButton_11.setText(_translate("MainWindow", "Save as DOC"))
        #self..setText(_translate("MainWindow", ""))
        self.pushButton_9.setText(_translate("MainWindow", "Summarize"))
        self.menuImage_processing.setTitle(_translate("MainWindow", "Image processing"))
        self.menuText_classfication.setTitle(_translate("MainWindow", "Text processing"))
        self.menuOptions.setTitle(_translate("MainWindow", "Options"))
        self.actionFace_detect.setText(_translate("MainWindow", "Back to Home Screen"))
        self.actionFace_recognition.setText(_translate("MainWindow", "Face recognition"))
        self.actionDigits_recognition.setText(_translate("MainWindow", "Image to text"))
        self.actionOther_objects_detection.setText(_translate("MainWindow", "Banker's database"))
        self.actionHome.setText(_translate("MainWindow", "Home"))
        self.actionExit.setText(_translate("MainWindow", "Exit"))
        self.actionText_summarizer.setText(_translate("MainWindow", "Text summarizer"))
        self.actionText_classification.setText(_translate("MainWindow", "Email classification"))
        self.actionFace_detect.triggered.connect(self.showFaceDetect)
        self.pushButton.clicked.connect(self.chooseImage)
        self.pushButton_2.clicked.connect(self.detectFace)
        self.actionExit.triggered.connect(self.exit)
        self.actionHome.triggered.connect(self.home)
        self.actionFace_recognition.triggered.connect(self.recog_face)
        self.pushButton_3.clicked.connect(self.face_store)
        self.pushButton_4.clicked.connect(self.face_recognize)
        self.pushButton_5.clicked.connect(self.text_classification)
        self.actionText_classification.triggered.connect(self.text_class)
        self.actionOther_objects_detection.triggered.connect(self.banker_bdb)
        self.pushButton_6.clicked.connect(self.bank_db)
        self.pushButton_7.clicked.connect(self.choose_image)
        self.pushButton_8.clicked.connect(self.image_to_text)
        self.actionDigits_recognition.triggered.connect(self.digit_rec)
        self.actionText_summarizer.triggered.connect(self.text_summary)
        self.pushButton_9.clicked.connect(self.generate_summary)
        self.pushButton_10.clicked.connect(self.save_as_pdf)
        self.pushButton_11.clicked.connect(self.save_as_doc)
        

    
    def text_summary(self):
        self.frame_6.show()
        self.frame.show()
        self.frame_5.show()
        self.frame_2.hide()
        self.frame_3.hide()
        self.frame_4.hide()


    def digit_rec(self):
        self.frame_5.show()
        self.frame.show()
        self.frame_2.hide()
        self.frame_3.hide()
        self.frame_4.show()
        self.frame_6.hide()        
    
    def banker_bdb(self):
        self.frame_4.show()
        self.frame.show()
        self.frame_2.hide()
        self.frame_3.hide()
        self.frame_5.hide()
        self.frame_6.hide()

    def text_class(self):
        self.frame_3.show()
        self.frame.show()
        self.frame_2.show()
        self.frame_4.hide()
        self.frame_5.hide()
        self.frame_6.hide()

    def recog_face(self):
        self.frame.show()
        self.frame_2.show()
        self.frame_3.hide()
        self.frame_4.hide()
        self.frame_5.hide()
        self.frame_6.hide()

    def home(self):
        self.frame.hide()
        self.frame_2.hide()
        self.frame_3.hide()
        self.frame_4.hide()
        self.frame_5.hide()
        self.frame_6.hide()

    def exit(self):
        if os.path.exists("text.txt"):
            os.remove("text.txt")
        sys.exit()
        

    def showFaceDetect(self):
        self.frame.hide()
        self.frame_2.hide()
        self.frame_3.hide()
        self.frame_4.hide()
        self.frame_5.hide()
        self.frame_6.hide()
        
        #func to select image from local directory
    def choose_image(self):
        location = QtWidgets.QFileDialog.getOpenFileName(self, 'Select File')#select location

        loc = location[0]

        self.filePath = os.path.normpath(loc)#set path

        self.image_8 = QtGui.QPixmap(self.filePath)#select image

        self.image_8 = self.image_8.scaled(self.label_7.width(), self.label_7.height())#scale image

        self.label_7.setPixmap(self.image_8)#set image in label window




    def chooseImage(self):
        location = QtWidgets.QFileDialog.getOpenFileName(self,'Select File')#select location
        
        loc = location[0]
        
        
        self.filePath = os.path.normpath(loc)#set path
        
        self.image_1 = QtGui.QPixmap(self.filePath)#select image
        
        self.image_1 = self.image_1.scaled(self.label_2.width(),self.label_2.height())#scale image

        self.label_2.setPixmap(self.image_1)#set image in label window
    

    #algo used to detect face
    def detectFace(self):
        dataset = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")#import dataset
        
        image = cv2.imread(self.filePath , cv2.COLOR_BGR2GRAY)#change to grey
        
        faces = dataset.detectMultiScale(image , 1.3)#scale
        
        for x,y,w,h in faces:
            cv2.rectangle(image,(x,y),(x+w,y+h) , (255,0,0) , 4)#set width height
        cv2.imwrite('result.jpg' , image) #save as result.jpg
        
        self.image_2 = QtGui.QPixmap('result.jpg')#select image
        
        self.image_2 = self.image_2.scaled(self.label_3.width(),self.label_3.height())#scale

        self.label_3.setPixmap(self.image_2)#set image in label window
    #algo used to store face data
    def face_store(self):
        import cv2
        import numpy as np
        
        dataset = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')#import dataset
        cap = cv2.VideoCapture(0)

        data = []

        while True:
            ret, img = cap.read()
            if ret:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                face = dataset.detectMultiScale(gray)
                # print(img)
                for x, y, w, h in face:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 5)
                    face_comp = img[y:y + h, x:x + w, :]

                    fc = cv2.resize(face_comp, (50, 50))

                    if len(data) < 20:
                        data.append(fc)
                    # print(data)

                if cv2.waitKey(2) == 27 or len(data) >= 20:
                    break
                cv2.imshow('result', img)
            else:
                print("Some error")

        data = np.asarray(data)
        np.save('user.npy', data)

        cv2.destroyAllWindows()
        cap.release()
        
        
    def face_recognize(self):
        #LIVE RECOGNITION
        text = self.textEdit_19.toPlainText()
        cam = cv2.VideoCapture(0)
        face_cas = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        f_01 = np.load('user.npy').reshape((20, 50 * 50 * 3))  # face_1
        f_02 = np.load('user1.npy').reshape((20, 50 * 50 * 3))  # face_2
        
        
        names = {
            0: text ,
            1: "akshit",
          
        }
        
        labels = np.zeros((40, 1))
        labels[:20, :] = 0.0  # first 20 for user_1 (0)
        labels[20:, :] = 1.0  # next 20 for user_2 (1)
        
        # combine all info into one data array
        data = np.concatenate([f_01,f_02])  # (60, 7500)
        print(data.shape, labels.shape)  # (60, 1)
        
        def distance(x1, x2):
            return np.sqrt(((x1 - x2) ** 2).sum())
        
        def knn(x, train, targets, k=5):
            m = train.shape[0]
            dist = []
            for ix in range(m):
                # compute distance from each point and store in dist
                dist.append(distance(x, train[ix]))
            dist = np.asarray(dist)
            indx = np.argsort(dist)
            print("Index...",indx)
            sorted_labels = labels[indx][:k]
            print("Sorted...",sorted_labels)
            counts = np.unique(sorted_labels, return_counts=True)
            print("Count...",counts)
            return counts[0][np.argmax(counts[1])]
        
        
        while True:
            ret, frame = cam.read()
            if ret == True:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cas.detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in faces:
                    face_component = frame[y:y + h, x:x + w, :]
                    fc = cv2.resize(face_component, (50, 50))
        
                    lab = knn(fc.flatten(), data, labels)
                    text = names[int(lab)]
                    cv2.putText(frame, text, (x, y), font, 1, (255, 255, 0), 2)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.imshow('face recognition', frame)
                k = cv2.waitKey(33) & 0xFF
                if k == 27:
                    break
            else:
                print('Error')
        
        cam.release()
        cv2.destroyAllWindows()

    def bank_db(self):
        


        def which_selected():
            print("At {0}".format(select.curselection()))
            return int(select.curselection()[0])
        
        
        def add_entry():
            phonelist.append([fnamevar.get(), lnamevar.get(), phonevar.get()])
            set_select()
        
        
        def update_entry():
            phonelist[which_selected()] = [fnamevar.get(),
                                           lnamevar.get(),
                                           phonevar.get()]
        
        
        def delete_entry():
            del phonelist[which_selected()]
            set_select()
        
        
        def load_entry():
            fname, lname, phone = phonelist[which_selected()]
            fnamevar.set(fname)
            lnamevar.set(lname)
            phonevar.set(phone)
        
        
        def make_window():
            global fnamevar, lnamevar, phonevar, select
            win = Tk()
        
            frame1 = Frame(win)
            frame1.pack()
        
            Label(frame1, text="First Name").grid(row=0, column=0, sticky=W)
            fnamevar = StringVar()
            fname = Entry(frame1, textvariable=fnamevar)
            fname.grid(row=0, column=1, sticky=W)
        
            Label(frame1, text="Last Name").grid(row=1, column=0, sticky=W)
            lnamevar = StringVar()
            lname = Entry(frame1, textvariable=lnamevar)
            lname.grid(row=1, column=1, sticky=W)
        
            Label(frame1, text="Phone").grid(row=2, column=0, sticky=W)
            phonevar = StringVar()
            phone = Entry(frame1, textvariable=phonevar)
            phone.grid(row=2, column=1, sticky=W)
        
            frame2 = Frame(win)       # Row of buttons
            frame2.pack()
            b1 = Button(frame2, text=" Add  ", command=add_entry)
            b2 = Button(frame2, text="Update", command=update_entry)
            b3 = Button(frame2, text="Delete", command=delete_entry)
            b4 = Button(frame2, text="Load  ", command=load_entry)
            b5 = Button(frame2, text="Refresh", command=set_select)
            b1.pack(side=LEFT)
            b2.pack(side=LEFT)
            b3.pack(side=LEFT)
            b4.pack(side=LEFT)
            b5.pack(side=LEFT)
        
            frame3 = Frame(win)       # select of names
            frame3.pack()
            scroll = Scrollbar(frame3, orient=VERTICAL)
            select = Listbox(frame3, yscrollcommand=scroll.set, height=6)
            scroll.config(command=select.yview)
            scroll.pack(side=RIGHT, fill=Y)
            select.pack(side=LEFT, fill=BOTH, expand=1)
            return win
        
        
        def set_select():
            phonelist.sort(key=lambda record: record[1])
            select.delete(0, END)
            for fname, lname, phone in phonelist:
                select.insert(END, "{0}, {1}".format(lname, fname))
        
        
        win = make_window()
        set_select()
        win.mainloop()


    def image_to_text(self):
        from scipy import misc        
        pytesseract.pytesseract.tesseract_cmd=r"C:\Program Files\Tesseract-OCR\tesseract.exe"#set path of module        
        img = misc.imread(self.filePath)#select image from local directory
        texts = pytesseract.image_to_string(img)#convert image to string        
        self.textEdit_9.setText(texts)#print text in text edit provided
        with open("text.txt", "w") as f:
            f.write(texts)
        f.close()
        
    def save_as_pdf(self):
        if os.path.exists("text.txt"):
            root = Tk()
            root.withdraw()
            files = [('PDF Files', '*.pdf')]
            file = asksaveasfile(filetypes = files, defaultextension = files)
            print (file.name)
            root.destroy()
            f = open("text.txt", "r")
            texts = f.read()
            pdf = FPDF(orientation = 'L')
            pdf.add_page()
            pdf.set_xy(0, 0)
            pdf.set_font('arial', 'B', 13.0)
            pdf.multi_cell(0, 6, txt=texts, border=0)
            pdf.output(file.name, 'F')
        else:
            pass
        
        
    def save_as_doc(self):
        if os.path.exists("text.txt"):
            root = Tk()
            root.withdraw()
            files = [('Word file', '*.docx')]
            file = asksaveasfile(filetypes = files, defaultextension = files)
            print (file.name)
            root.destroy()
            f = open("text.txt", "r")
            texts = f.read()
            document = Document()
            document.add_paragraph(texts)
            document.add_page_break()
            document.save(file.name)
        else:
            pass

    def generate_summary(self):
        text = self.textEdit_3.toPlainText()
        words = word_tokenize(text)
        sentences = sent_tokenize(text)
        sWords = set(stopwords.words("english"))
        w_net = WordNetLemmatizer()

        freq_dist = dict()

        for word in words:
            word = word.lower()

            if word in sWords:
                continue

            word = w_net.lemmatize(word, pos='v')

            if word in freq_dist:
                freq_dist[word] += 1
            else:
                freq_dist[word] = 1

        sent_dist = dict()

        for sentence in sentences:
            for word, freq in freq_dist.items():
                if word in sentence:
                    if sentence in sent_dist:
                        #print("Word =>", word)
                        #print("Sentence =>", sentence)
                        sent_dist[sentence] += freq
                    else:
                        #print("Word =>", word)
                        #print("Sentence =>", sentence)
                        sent_dist[sentence] = freq

        avg = int(sum(sent_dist.values()) / len(sent_dist))
        summary = ""
        for sentence in sentences:
            if sent_dist[sentence] > avg * 1.1:
                #         print(sentence)
                summary += " " + sentence

        self.textEdit_4.setText(summary)


    def text_classification(self):
        text = self.textEdit_2.toPlainText()
        mails = pd.read_csv('spam.csv', encoding = 'latin-1')
        mails.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1, inplace = True)
        mails.rename(columns = {'v1': 'labels', 'v2': 'message'}, inplace = True)
        mails['label'] = mails['labels'].map({'ham': 0, 'spam': 1})
        mails.drop(['labels'], axis = 1, inplace = True)
        
        totalMails = 4825 + 747
        trainIndex, testIndex = list(), list()
        for i in range(mails.shape[0]):
            if np.random.uniform(0, 1) < 0.75:
                trainIndex += [i]
            else:
                testIndex += [i]
        trainData = mails.loc[trainIndex]
        testData = mails.loc[testIndex]
        
        trainData.reset_index(inplace = True)
        trainData.drop(['index'], axis = 1, inplace = True)
        
        
        testData.reset_index(inplace = True)
        testData.drop(['index'], axis = 1, inplace = True)

        def process_message(message, lower_case = True, stem = True, stop_words = True, gram = 2):
            if lower_case:
                message = message.lower()
            words = word_tokenize(message)
            words = [w for w in words if len(w) > 2]
            if gram > 1:
                w = []
                for i in range(len(words) - gram + 1):
                    w += [' '.join(words[i:i + gram])]
                return w
            if stop_words:
                sw = stopwords.words('english')
                words = [word for word in words if word not in sw]
            if stem:
                stemmer = PorterStemmer()
                words = [stemmer.stem(word) for word in words]   
            return words
        
        
        class SpamClassifier(object):
            def __init__(self, trainData, method = 'tf-idf'):
                self.mails, self.labels = trainData['message'], trainData['label']
                self.method = method
        
            def train(self):
                self.calc_TF_and_IDF()
                if self.method == 'tf-idf':
                    self.calc_TF_IDF()
                else:
                    self.calc_prob()
        
            def calc_prob(self):
                self.prob_spam = dict()
                self.prob_ham = dict()
                for word in self.tf_spam:
                    self.prob_spam[word] = (self.tf_spam[word] + 1) / (self.spam_words + \
                                                                        len(list(self.tf_spam.keys())))
                for word in self.tf_ham:
                    self.prob_ham[word] = (self.tf_ham[word] + 1) / (self.ham_words + \
                                                                        len(list(self.tf_ham.keys())))
                self.prob_spam_mail, self.prob_ham_mail = self.spam_mails / self.total_mails, self.ham_mails / self.total_mails 
        
        
            def calc_TF_and_IDF(self):
                noOfMessages = self.mails.shape[0]
                self.spam_mails, self.ham_mails = self.labels.value_counts()[1], self.labels.value_counts()[0]
                self.total_mails = self.spam_mails + self.ham_mails
                self.spam_words = 0
                self.ham_words = 0
                self.tf_spam = dict()
                self.tf_ham = dict()
                self.idf_spam = dict()
                self.idf_ham = dict()
                for i in range(noOfMessages):
                    message_processed = process_message(self.mails[i])
                    count = list() #To keep track of whether the word has ocured in the message or not.
                                   #For IDF
                    for word in message_processed:
                        if self.labels[i]:
                            self.tf_spam[word] = self.tf_spam.get(word, 0) + 1
                            self.spam_words += 1
                        else:
                            self.tf_ham[word] = self.tf_ham.get(word, 0) + 1
                            self.ham_words += 1
                        if word not in count:
                            count += [word]
                    for word in count:
                        if self.labels[i]:
                            self.idf_spam[word] = self.idf_spam.get(word, 0) + 1
                        else:
                            self.idf_ham[word] = self.idf_ham.get(word, 0) + 1
        
            def calc_TF_IDF(self):
                self.prob_spam = dict()
                self.prob_ham = dict()
                self.sum_tf_idf_spam = 0
                self.sum_tf_idf_ham = 0
                for word in self.tf_spam:
                    self.prob_spam[word] = (self.tf_spam[word]) * log((self.spam_mails + self.ham_mails) \
                                                                  / (self.idf_spam[word] + self.idf_ham.get(word, 0)))
                    self.sum_tf_idf_spam += self.prob_spam[word]
                for word in self.tf_spam:
                    self.prob_spam[word] = (self.prob_spam[word] + 1) / (self.sum_tf_idf_spam + len(list(self.prob_spam.keys())))
                    
                for word in self.tf_ham:
                    self.prob_ham[word] = (self.tf_ham[word]) * log((self.spam_mails + self.ham_mails) \
                                                                  / (self.idf_spam.get(word, 0) + self.idf_ham[word]))
                    self.sum_tf_idf_ham += self.prob_ham[word]
                for word in self.tf_ham:
                    self.prob_ham[word] = (self.prob_ham[word] + 1) / (self.sum_tf_idf_ham + len(list(self.prob_ham.keys())))
                    
            
                self.prob_spam_mail, self.prob_ham_mail = self.spam_mails / self.total_mails, self.ham_mails / self.total_mails 
                            
            def classify(self, processed_message):
                pSpam, pHam = 0, 0
                for word in processed_message:                
                    if word in self.prob_spam:
                        pSpam += log(self.prob_spam[word])
                    else:
                        if self.method == 'tf-idf':
                            pSpam -= log(self.sum_tf_idf_spam + len(list(self.prob_spam.keys())))
                        else:
                            pSpam -= log(self.spam_words + len(list(self.prob_spam.keys())))
                    if word in self.prob_ham:
                        pHam += log(self.prob_ham[word])
                    else:
                        if self.method == 'tf-idf':
                            pHam -= log(self.sum_tf_idf_ham + len(list(self.prob_ham.keys()))) 
                        else:
                            pHam -= log(self.ham_words + len(list(self.prob_ham.keys())))
                    pSpam += log(self.prob_spam_mail)
                    pHam += log(self.prob_ham_mail)
                return pSpam >= pHam
            
            def predict(self, testData):
                result = dict()
                for (i, message) in enumerate(testData):
                    processed_message = process_message(message)
                    result[i] = int(self.classify(processed_message))
                return result
        
        
        
        
        sc_tf_idf = SpamClassifier(trainData, 'tf-idf')
        sc_tf_idf.train()
        preds_tf_idf = sc_tf_idf.predict(testData['message'])
        #metrics(testData['label'], preds_tf_idf)
        
        
        sc_bow = SpamClassifier(trainData, 'bow')
        sc_bow.train()
        preds_bow = sc_bow.predict(testData['message'])
        #metrics(testData['label'], preds_bow)
        
        pm = process_message(text)
        k = sc_tf_idf.classify(pm)
        if k:
            self.textEdit.setText("spam")
        else:
            self.textEdit.setText("ham")
        
       

        


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
