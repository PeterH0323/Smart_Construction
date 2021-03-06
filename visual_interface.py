# -*- coding: utf-8 -*-
# @Time    : 2021/3/6 15:36
# @Author  : PeterH
# @Email   : peterhuang0323@outlook.com
# @File    : visual_interface.py
# @Software: PyCharm
# @Brief   :
import time
from pathlib import Path

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import QApplication, QMainWindow, \
    QFileDialog, QWidget  # QMainWindow, QApplication, QDialog, QWidget, QMessageBox
import sys

from UI.main_window import Ui_MainWindow
from detect_visual import YOLOPredict

CODE_VER = "V0.1"


class PredictDataHandlerThread(QThread):
    predict_message_trigger = pyqtSignal(str)

    def __init__(self, output_player):
        super(PredictDataHandlerThread, self).__init__()
        self.running = False

        '''加载模型'''
        self.project_root = Path.cwd()
        self.parameter_agnostic_nms = False
        self.parameter_augment = False
        self.parameter_classes = None
        self.parameter_conf_thres = 0.4
        self.parameter_device = ''
        self.parameter_img_size = 640
        self.parameter_iou_thres = 0.5
        self.parameter_output = self.project_root.joinpath(r'inference/output')
        self.parameter_save_txt = False
        self.parameter_source = ''
        self.parameter_update = False
        self.parameter_view_img = False
        self.parameter_weights = ['./weights/helmet_head_person_m.pt']
        self.predict_model = YOLOPredict(self.parameter_device, self.parameter_weights, self.parameter_img_size)

        # 播放插件
        self.output_player = output_player

    def __del__(self):
        self.running = False
        self.wait()

    def run(self):
        self.running = True
        while self.running:
            # if self.predict_model.predict_info != "":
            self.predict_message_trigger.emit(self.predict_model.predict_info)
            self.predict_model.predict_info = ""
            time.sleep(0.01)

    def thread_started(self):
        print("PredictDataHandlerThread started")

        output_predict_file = self.predict_model.detect(self.parameter_output,
                                                        self.parameter_source,
                                                        self.parameter_view_img,
                                                        self.parameter_save_txt,
                                                        self.parameter_img_size,
                                                        self.parameter_augment,
                                                        self.parameter_conf_thres,
                                                        self.parameter_iou_thres,
                                                        self.parameter_classes,
                                                        self.parameter_agnostic_nms,
                                                        self.parameter_update)

        # 将 str 路径转为 QUrl 并显示
        self.output_player.setMedia(QMediaContent(QUrl.fromLocalFile(output_predict_file)))  # 选取视频文件
        self.output_player.pause()  # 显示媒体

        # 停止线程
        self.running = False

    def thread_finished(self):
        print("PredictDataHandlerThread finished")


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("VAT ROLL COMPARE LABEL TOOL" + " " + CODE_VER)
        self.showMaximized()

        '''按键绑定'''
        # 输入媒体
        self.import_media_pushButton.clicked.connect(self.import_media)  # 导入
        self.start_predict_pushButton.clicked.connect(self.predict_button_click)  # 开始推理
        # 输出媒体
        # self.open_predict_file_pushButton.clicked.connect(self.play_pause_button_click)  # 文件中显示推理视频
        # 下方
        self.play_pushButton.clicked.connect(self.play_pause_button_click)  # 播放
        self.pause_pushButton.clicked.connect(self.play_pause_button_click)  # 暂停

        '''媒体流绑定输出'''
        self.input_player = QMediaPlayer()  # 媒体输入的widget
        self.input_player.setVideoOutput(self.input_video_widget)
        self.input_player.positionChanged.connect(self.change_slide_bar)  # 播放进度条

        self.output_player = QMediaPlayer()  # 媒体输出的widget
        self.output_player.setVideoOutput(self.output_video_widget)

        # 播放时长, 以 input 的时长为准
        self.video_length = 0

        # 推理使用另外一线程
        self.predict_data_handler_thread = PredictDataHandlerThread(self.output_player)
        self.predict_data_handler_thread.predict_message_trigger.connect(self.add_messages)
        self.predict_data_handler_thread.started.connect(self.predict_data_handler_thread.thread_started)
        self.predict_data_handler_thread.finished.connect(self.predict_data_handler_thread.thread_finished)

    @pyqtSlot(str)
    def add_messages(self, message):
        if message != "":
            self.predict_info_plainTextEdit.appendPlainText(message)

    def import_media(self):
        """
        导入媒体文件
        :return:
        """
        self.parameter_source = QFileDialog.getOpenFileUrl()[0]
        self.input_player.setMedia(QMediaContent(self.parameter_source))  # 选取视频文件

        # 设置 output 为一张图片，防止资源被占用
        path_current = str(Path.cwd().joinpath("area_dangerous\1.jpg"))
        self.output_player.setMedia(QMediaContent(QUrl.fromLocalFile(path_current)))

        # 将 QUrl 路径转为 本地路径str
        self.predict_data_handler_thread.parameter_source = self.parameter_source.toLocalFile()
        self.input_player.pause()  # 显示媒体
        # self.output_player.setMedia(QMediaContent(QFileDialog.getOpenFileUrl()[0]))  # 选取视频文件

    def predict_button_click(self):
        self.predict_data_handler_thread.start()

    def change_slide_bar(self, position):
        """
        进度条移动
        :param position:
        :return:
        """
        self.video_length = self.input_player.duration() + 0.1
        self.video_horizontalSlider.setValue(round((position / self.video_length) * 100))
        self.video_percent_label.setText(str(round((position / self.video_length) * 100, 2)) + '%')

    @pyqtSlot()
    def play_pause_button_click(self):
        """
        播放、暂停按钮回调事件
        :return:
        """
        name = self.sender().objectName()

        if self.parameter_source == "":
            return

        if name == "play_pushButton":
            print("play")
            self.input_player.play()
            self.output_player.play()

        elif name == "pause_pushButton":
            self.input_player.pause()
            self.output_player.pause()

    @pyqtSlot()
    def closeEvent(self, *args, **kwargs):
        """
        重写关闭事件
        :param args:
        :param kwargs:
        :return:
        """
        print("Close")


if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
