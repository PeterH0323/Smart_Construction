# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main_window.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1217, 933)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(80, 150, 511, 491))
        self.groupBox.setObjectName("groupBox")
        self.input_video_widget = QtWidgets.QWidget(self.groupBox)
        self.input_video_widget.setGeometry(QtCore.QRect(20, 20, 451, 431))
        self.input_video_widget.setObjectName("input_video_widget")
        self.pushButton = QtWidgets.QPushButton(self.groupBox)
        self.pushButton.setGeometry(QtCore.QRect(60, 460, 75, 23))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_2.setGeometry(QtCore.QRect(170, 460, 75, 23))
        self.pushButton_2.setObjectName("pushButton_2")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(630, 140, 531, 511))
        self.groupBox_2.setObjectName("groupBox_2")
        self.output_video_widget = QtWidgets.QWidget(self.groupBox_2)
        self.output_video_widget.setGeometry(QtCore.QRect(20, 20, 451, 431))
        self.output_video_widget.setObjectName("output_video_widget")
        self.pushButton_5 = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_5.setGeometry(QtCore.QRect(100, 470, 75, 23))
        self.pushButton_5.setObjectName("pushButton_5")
        self.play_pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.play_pushButton.setGeometry(QtCore.QRect(520, 700, 75, 23))
        self.play_pushButton.setObjectName("play_pushButton")
        self.pause_pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pause_pushButton.setGeometry(QtCore.QRect(620, 700, 75, 23))
        self.pause_pushButton.setObjectName("pause_pushButton")
        self.predict_info_plainTextEdit = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.predict_info_plainTextEdit.setGeometry(QtCore.QRect(140, 730, 951, 81))
        self.predict_info_plainTextEdit.setObjectName("predict_info_plainTextEdit")
        self.predict_progressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.predict_progressBar.setGeometry(QtCore.QRect(270, 110, 118, 23))
        self.predict_progressBar.setProperty("value", 24)
        self.predict_progressBar.setObjectName("predict_progressBar")
        self.horizontalSlider = QtWidgets.QSlider(self.centralwidget)
        self.horizontalSlider.setGeometry(QtCore.QRect(530, 660, 160, 22))
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setObjectName("horizontalSlider")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(710, 660, 61, 21))
        self.label.setObjectName("label")
        self.author_label = QtWidgets.QLabel(self.centralwidget)
        self.author_label.setGeometry(QtCore.QRect(970, 90, 91, 31))
        self.author_label.setObjectName("author_label")
        self.gpu_info_label = QtWidgets.QLabel(self.centralwidget)
        self.gpu_info_label.setGeometry(QtCore.QRect(530, 830, 271, 31))
        self.gpu_info_label.setObjectName("gpu_info_label")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1217, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox.setTitle(_translate("MainWindow", "输入视频"))
        self.pushButton.setText(_translate("MainWindow", "导入"))
        self.pushButton_2.setText(_translate("MainWindow", "推理"))
        self.groupBox_2.setTitle(_translate("MainWindow", "输出视频"))
        self.pushButton_5.setText(_translate("MainWindow", "打开文件"))
        self.play_pushButton.setText(_translate("MainWindow", "播放"))
        self.pause_pushButton.setText(_translate("MainWindow", "暂停"))
        self.label.setText(_translate("MainWindow", "0 %"))
        self.author_label.setText(_translate("MainWindow", "黄显钧 出品"))
        self.gpu_info_label.setText(_translate("MainWindow", "TextLabel"))
