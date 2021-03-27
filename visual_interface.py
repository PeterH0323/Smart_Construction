# -*- coding: utf-8 -*-
# @Time    : 2021/3/6 15:36
# @Author  : PeterH
# @Email   : peterhuang0323@outlook.com
# @File    : visual_interface.py
# @Software: PyCharm
# @Brief   :

# 解决 exe 打包 Can't get source for 的问题 start ======
# https://github.com/pytorch/vision/issues/1899#issuecomment-598200938
import torch.jit


def script_method(fn, _rcb=None):
    return fn


def script(obj, optimize=True, _frames_up=0, _rcb=None):
    return obj


torch.jit.script_method = script_method
torch.jit.script = script
# 解决 exe 打包 Can't get source for 的问题 end ======

import os
import time
import sys
from pathlib import Path
from GPUtil import GPUtil
from PyQt5.QtCore import QThread, pyqtSignal, QUrl, pyqtSlot, QTimer, QDateTime, Qt
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtGui import QColor, QBrush, QIcon, QPixmap
from PyQt5.QtChart import QDateTimeAxis, QValueAxis, QSplineSeries, QChart
import torch
from UI.main_window import Ui_MainWindow
from detect_visual import YOLOPredict
from utils.datasets import img_formats

CODE_VER = "V2.0"
PREDICT_SHOW_TAB_INDEX = 0
REAL_TIME_PREDICT_TAB_INDEX = 1


def get_gpu_info():
    """
    获取 GPU 信息
    :return:
    """

    gpu_list = []
    # GPUtil.showUtilization()

    # 获取多个GPU的信息，存在列表里
    for gpu in GPUtil.getGPUs():
        # print('gpu.id:', gpu.id)
        # print('GPU总量：', gpu.memoryTotal)
        # print('GPU使用量：', gpu.memoryUsed)
        # print('gpu使用占比:', gpu.memoryUtil * 100)  # 内存使用率
        # print('gpu load:', gpu.load * 100)  # 使用率
        # 按GPU逐个添加信息
        gpu_list.append({"gpu_id": gpu.id,
                         "gpu_memoryTotal": gpu.memoryTotal,
                         "gpu_memoryUsed": gpu.memoryUsed,
                         "gpu_memoryUtil": gpu.memoryUtil * 100,
                         "gpu_load": gpu.load * 100})

    return gpu_list


class PredictDataHandlerThread(QThread):
    """
    打印信息的线程
    """
    predict_message_trigger = pyqtSignal(str)

    def __init__(self, predict_model):
        super(PredictDataHandlerThread, self).__init__()
        self.running = False
        self.predict_model = predict_model

    def __del__(self):
        self.running = False
        # self.destroyed()

    def run(self):
        self.running = True
        over_time = 0
        while self.running:
            if self.predict_model.predict_info != "":
                self.predict_message_trigger.emit(self.predict_model.predict_info)
                self.predict_model.predict_info = ""
                over_time = 0
            time.sleep(0.01)
            over_time += 1

            if over_time > 100000:
                self.running = False


class PredictHandlerThread(QThread):
    """
    进行模型推理的线程
    """

    def __init__(self, input_player, output_player, out_file_path, weight_path,
                 predict_info_plain_text_edit, predict_progress_bar, fps_label,
                 button_dict, input_tab, output_tab, input_image_label, output_image_label,
                 real_time_show_predict_flag):
        super(PredictHandlerThread, self).__init__()
        self.running = False

        '''加载模型'''
        self.predict_model = YOLOPredict(weight_path, out_file_path)
        self.output_predict_file = ""
        self.parameter_source = ''

        # 传入的QT插件
        self.input_player = input_player
        self.output_player = output_player
        self.predict_info_plainTextEdit = predict_info_plain_text_edit
        self.predict_progressBar = predict_progress_bar
        self.fps_label = fps_label
        self.button_dict = button_dict
        self.input_tab = input_tab
        self.output_tab = output_tab
        self.input_image_label = input_image_label
        self.output_image_label = output_image_label

        # 是否实时显示推理图片
        self.real_time_show_predict_flag = real_time_show_predict_flag

        # 创建显示进程
        self.predict_data_handler_thread = PredictDataHandlerThread(self.predict_model)
        self.predict_data_handler_thread.predict_message_trigger.connect(self.add_messages)

    def __del__(self):
        self.running = False
        # self.destroyed()

    def run(self):
        self.predict_data_handler_thread.start()

        self.predict_progressBar.setValue(0)  # 进度条清零
        for item, button in self.button_dict.items():
            button.setEnabled(False)

        image_flag = os.path.splitext(self.parameter_source)[-1].lower() in img_formats
        qt_input = None
        qt_output = None

        if not image_flag and self.real_time_show_predict_flag:
            qt_input = self.input_image_label
            qt_output = self.output_image_label
            # tab 设置显示第二栏
            self.input_tab.setCurrentIndex(REAL_TIME_PREDICT_TAB_INDEX)
            self.output_tab.setCurrentIndex(REAL_TIME_PREDICT_TAB_INDEX)

        with torch.no_grad():
            self.output_predict_file = self.predict_model.detect(self.parameter_source,
                                                                 qt_input=qt_input,
                                                                 qt_output=qt_output)

        if self.output_predict_file != "":
            # 将 str 路径转为 QUrl 并显示
            self.input_player.setMedia(QMediaContent(QUrl.fromLocalFile(self.parameter_source)))  # 选取视频文件
            self.input_player.pause()  # 显示媒体

            self.output_player.setMedia(QMediaContent(QUrl.fromLocalFile(self.output_predict_file)))  # 选取视频文件
            self.output_player.pause()  # 显示媒体

            # tab 设置显示第一栏
            self.input_tab.setCurrentIndex(PREDICT_SHOW_TAB_INDEX)
            self.output_tab.setCurrentIndex(PREDICT_SHOW_TAB_INDEX)

            # video_flag = os.path.splitext(self.parameter_source)[-1].lower() in vid_formats
            for item, button in self.button_dict.items():
                if image_flag and item in ['play_pushButton', 'pause_pushButton']:
                    continue
                button.setEnabled(True)
        # self.predict_data_handler_thread.running = False

    @pyqtSlot(str)
    def add_messages(self, message):
        if message != "":
            self.predict_info_plainTextEdit.appendPlainText(message)

            if ":" not in message:
                # 跳过无用字符
                return

            split_message = message.split(" ")

            # 设置进度条
            if "video" in message:
                percent = split_message[2][1:-1].split("/")  # 提取图片的序号
                value = int((int(percent[0]) / int(percent[1])) * 100)
                value = value if (int(percent[1]) - int(percent[0])) > 2 else 100
                self.predict_progressBar.setValue(value)
            else:
                self.predict_progressBar.setValue(100)

            # 设置 FPS
            second_count = 1 / float(split_message[-1][1:-2])
            self.fps_label.setText(f"--> {second_count:.1f} FPS")


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, weight_path, out_file_path, real_time_show_predict_flag: bool, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("Intelligent Monitoring System of Construction Site Software " + CODE_VER)
        self.showMaximized()

        '''按键绑定'''
        # 输入媒体
        self.import_media_pushButton.clicked.connect(self.import_media)  # 导入
        self.start_predict_pushButton.clicked.connect(self.predict_button_click)  # 开始推理
        # 输出媒体
        self.open_predict_file_pushButton.clicked.connect(self.open_file_in_browser)  # 文件中显示推理视频
        # 下方
        self.play_pushButton.clicked.connect(self.play_pause_button_click)  # 播放
        self.pause_pushButton.clicked.connect(self.play_pause_button_click)  # 暂停
        self.button_dict = dict()
        self.button_dict.update({"import_media_pushButton": self.import_media_pushButton,
                                 "start_predict_pushButton": self.start_predict_pushButton,
                                 "open_predict_file_pushButton": self.open_predict_file_pushButton,
                                 "play_pushButton": self.play_pushButton,
                                 "pause_pushButton": self.pause_pushButton,
                                 "real_time_checkBox": self.real_time_checkBox
                                 })

        '''媒体流绑定输出'''
        self.input_player = QMediaPlayer()  # 媒体输入的widget
        self.input_player.setVideoOutput(self.input_video_widget)
        self.input_player.positionChanged.connect(self.change_slide_bar)  # 播放进度条

        self.output_player = QMediaPlayer()  # 媒体输出的widget
        self.output_player.setVideoOutput(self.output_video_widget)

        '''初始化GPU chart'''
        self.series = QSplineSeries()
        self.chart_init()

        '''初始化GPU定时查询定时器'''
        # 使用QTimer，0.5秒触发一次，更新数据
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.draw_gpu_info_chart)
        self.timer.start(1000)

        # 播放时长, 以 input 的时长为准
        self.video_length = 0
        self.out_file_path = out_file_path
        # 推理使用另外一线程
        self.predict_handler_thread = PredictHandlerThread(self.input_player,
                                                           self.output_player,
                                                           self.out_file_path,
                                                           weight_path,
                                                           self.predict_info_plainTextEdit,
                                                           self.predict_progressBar,
                                                           self.fps_label,
                                                           self.button_dict,
                                                           self.input_media_tabWidget,
                                                           self.output_media_tabWidget,
                                                           self.input_real_time_label,
                                                           self.output_real_time_label,
                                                           real_time_show_predict_flag
                                                           )
        self.weight_label.setText(f" Using weight : ****** {Path(weight_path[0]).name} ******")
        # 界面美化
        self.gen_better_gui()

        self.media_source = ""  # 推理媒体的路径

        self.predict_progressBar.setValue(0)  # 进度条归零

        '''check box 绑定'''
        self.real_time_checkBox.stateChanged.connect(self.real_time_checkbox_state_changed)
        self.real_time_checkBox.setChecked(real_time_show_predict_flag)
        self.real_time_check_state = self.real_time_checkBox.isChecked()

    def gen_better_gui(self):
        """
        美化界面
        :return:
        """
        # Play 按钮
        play_icon = QIcon()
        play_icon.addPixmap(QPixmap("./UI/icon/play.png"), QIcon.Normal, QIcon.Off)
        self.play_pushButton.setIcon(play_icon)

        # Pause 按钮
        play_icon = QIcon()
        play_icon.addPixmap(QPixmap("./UI/icon/pause.png"), QIcon.Normal, QIcon.Off)
        self.pause_pushButton.setIcon(play_icon)

        # 隐藏 tab 标题栏
        self.input_media_tabWidget.tabBar().hide()
        self.output_media_tabWidget.tabBar().hide()
        # tab 设置显示第一栏
        self.input_media_tabWidget.setCurrentIndex(PREDICT_SHOW_TAB_INDEX)
        self.output_media_tabWidget.setCurrentIndex(PREDICT_SHOW_TAB_INDEX)

        # 设置显示图片的 label 为黑色背景
        self.input_real_time_label.setStyleSheet("QLabel{background:black}")
        self.output_real_time_label.setStyleSheet("QLabel{background:black}")

    def real_time_checkbox_state_changed(self):
        """
        切换是否实时显示推理图片
        :return:
        """
        self.real_time_check_state = self.real_time_checkBox.isChecked()
        self.predict_handler_thread.real_time_show_predict_flag = self.real_time_check_state

    def chart_init(self):
        """
        初始化 GPU 折线图
        :return:
        """
        # self.gpu_info_chart._chart = QChart(title="折线图堆叠")  # 创建折线视图
        self.gpu_info_chart._chart = QChart()  # 创建折线视图
        # chart._chart.setBackgroundVisible(visible=False)      # 背景色透明
        self.gpu_info_chart._chart.setBackgroundBrush(QBrush(QColor("#FFFFFF")))  # 改变图背景色

        # 设置曲线名称
        self.series.setName("GPU Utilization")
        # 把曲线添加到QChart的实例中
        self.gpu_info_chart._chart.addSeries(self.series)
        # 声明并初始化X轴，Y轴
        self.dtaxisX = QDateTimeAxis()
        self.vlaxisY = QValueAxis()
        # 设置坐标轴显示范围
        self.dtaxisX.setMin(QDateTime.currentDateTime().addSecs(-300 * 1))
        self.dtaxisX.setMax(QDateTime.currentDateTime().addSecs(0))
        self.vlaxisY.setMin(0)
        self.vlaxisY.setMax(100)
        # 设置X轴时间样式
        self.dtaxisX.setFormat("hh:mm:ss")
        # 设置坐标轴上的格点
        self.dtaxisX.setTickCount(5)
        self.vlaxisY.setTickCount(10)
        # 设置坐标轴名称
        self.dtaxisX.setTitleText("Time")
        self.vlaxisY.setTitleText("Percent")
        # 设置网格不显示
        self.vlaxisY.setGridLineVisible(False)
        # 把坐标轴添加到chart中
        self.gpu_info_chart._chart.addAxis(self.dtaxisX, Qt.AlignBottom)
        self.gpu_info_chart._chart.addAxis(self.vlaxisY, Qt.AlignLeft)
        # 把曲线关联到坐标轴
        self.series.attachAxis(self.dtaxisX)
        self.series.attachAxis(self.vlaxisY)
        # 生成 折线图
        self.gpu_info_chart.setChart(self.gpu_info_chart._chart)

    def draw_gpu_info_chart(self):
        """
        绘制 GPU 折线图
        :return:
        """
        # 获取当前时间
        time_current = QDateTime.currentDateTime()
        # 更新X轴坐标
        self.dtaxisX.setMin(QDateTime.currentDateTime().addSecs(-300 * 1))
        self.dtaxisX.setMax(QDateTime.currentDateTime().addSecs(0))
        # 当曲线上的点超出X轴的范围时，移除最早的点
        remove_count = 600
        if self.series.count() > remove_count:
            self.series.removePoints(0, self.series.count() - remove_count)
        # 对 y 赋值
        # yint = random.randint(0, 100)
        gpu_info = get_gpu_info()
        yint = gpu_info[0].get("gpu_load")
        # 添加数据到曲线末端
        self.series.append(time_current.toMSecsSinceEpoch(), yint)

    def import_media(self):
        """
        导入媒体文件
        :return:
        """
        self.media_source = QFileDialog.getOpenFileUrl()[0]
        self.input_player.setMedia(QMediaContent(self.media_source))  # 选取视频文件

        # 设置 output 为一张图片，防止资源被占用
        path_current = str(Path.cwd().joinpath("area_dangerous\1.jpg"))
        self.output_player.setMedia(QMediaContent(QUrl.fromLocalFile(path_current)))

        # 将 QUrl 路径转为 本地路径str
        self.predict_handler_thread.parameter_source = self.media_source.toLocalFile()
        self.input_player.pause()  # 显示媒体

        image_flag = os.path.splitext(self.predict_handler_thread.parameter_source)[-1].lower() in img_formats
        for item, button in self.button_dict.items():
            if image_flag and item in ['play_pushButton', 'pause_pushButton']:
                button.setEnabled(False)
            else:
                button.setEnabled(True)
        # self.output_player.setMedia(QMediaContent(QFileDialog.getOpenFileUrl()[0]))  # 选取视频文件

    def predict_button_click(self):
        """
        推理按钮
        :return:
        """
        # 启动线程去调用
        self.predict_handler_thread.start()

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

        if self.media_source == "":
            return

        if name == "play_pushButton":
            print("play")
            self.input_player.play()
            self.output_player.play()

        elif name == "pause_pushButton":
            self.input_player.pause()
            self.output_player.pause()

    @pyqtSlot()
    def open_file_in_browser(self):
        os.system(f"start explorer {self.out_file_path}")

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
    app = QApplication(sys.argv)

    weight_root = Path.cwd().joinpath("weights")
    if not weight_root.exists():
        raise FileNotFoundError("weights not found !!!")

    weight_file = [item for item in weight_root.iterdir() if item.suffix == ".pt"]
    weight_root = [str(weight_file[0])]  # 权重文件位置
    out_file_root = Path.cwd().joinpath(r'inference/output')
    out_file_root.parent.mkdir(exist_ok=True)
    out_file_root.mkdir(exist_ok=True)

    real_time_show_predict = True  # 是否实时显示推理图片，有可能导致卡顿，软件卡死

    main_window = MainWindow(weight_root, out_file_root, real_time_show_predict)

    # 设置窗口图标
    icon = QIcon()
    icon.addPixmap(QPixmap("./UI/icon/icon.ico"), QIcon.Normal, QIcon.Off)
    main_window.setWindowIcon(icon)

    main_window.show()
    sys.exit(app.exec_())
