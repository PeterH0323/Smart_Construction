# -*- coding: utf-8 -*-
# @Time    : 2021/3/6 15:36
# @Author  : PeterH
# @Email   : peterhuang0323@outlook.com
# @File    : visual_interface.py
# @Software: PyCharm
# @Brief   :
import datetime
import random
import time
from pathlib import Path

from GPUtil import GPUtil
from PyQt5.QtCore import *
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtGui import QPainter, QCursor, QColor, QBrush
from PyQt5.QtChart import QDateTimeAxis, QValueAxis, QSplineSeries, QChart, QChartView, QLineSeries, QCategoryAxis

from UI.main_window import Ui_MainWindow
from detect_visual import YOLOPredict

CODE_VER = "V0.1"


def get_gpu_info():
    """
    获取 GPU 信息
    :return:
    """

    gpu_list = []
    GPUtil.showUtilization()

    # 获取多个GPU的信息，存在列表里
    for gpu in GPUtil.getGPUs():
        print('gpu.id:', gpu.id)
        print('GPU总量：', gpu.memoryTotal)
        print('GPU使用量：', gpu.memoryUsed)
        print('gpu使用占比:', gpu.memoryUtil * 100)  # 内存使用率
        print('gpu load:', gpu.load * 100)  # 使用率
        # 按GPU逐个添加信息
        gpu_list.append([gpu.id, gpu.memoryTotal, gpu.memoryUsed, gpu.memoryUtil * 100])

    return gpu_list


#
# class GPUInfoHandlerThread(QThread):
#     """
#     打印GPU信息
#     """
#     gpu_message_trigger = pyqtSignal(str)
#
#     def __init__(self):
#         super(GPUInfoHandlerThread, self).__init__()
#         self.running = False
#
#     def __del__(self):
#         self.running = False
#         self.wait()
#
#     def run(self):
#         self.running = True
#         while self.running:
#             self.gpu_message_trigger.emit(self.predict_model.predict_info)
#             time.sleep(0.01)


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
        self.wait()

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

    def __init__(self, output_player, predict_info_plainTextEdit, predict_progressBar, fps_label):
        super(PredictHandlerThread, self).__init__()
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
        self.parameter_weights = ['./weights/helmet_head_person_m.pt']  # 权重文件路径，修改这里
        self.predict_model = YOLOPredict(self.parameter_device, self.parameter_weights, self.parameter_img_size)
        self.output_predict_file = ""
        # 传入的QT插件
        self.output_player = output_player
        self.predict_info_plainTextEdit = predict_info_plainTextEdit
        self.predict_progressBar = predict_progressBar
        self.fps_label = fps_label

        # 创建显示进程
        self.predict_data_handler_thread = PredictDataHandlerThread(self.predict_model)
        self.predict_data_handler_thread.predict_message_trigger.connect(self.add_messages)

    def __del__(self):
        self.running = False
        self.wait()

    def run(self):
        self.predict_data_handler_thread.start()

        self.predict_progressBar.setValue(0)  # 进度条清零

        self.output_predict_file = self.predict_model.detect(self.parameter_output,
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

        if self.output_predict_file != "":
            # 将 str 路径转为 QUrl 并显示
            self.output_player.setMedia(QMediaContent(QUrl.fromLocalFile(self.output_predict_file)))  # 选取视频文件
            self.output_player.pause()  # 显示媒体

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

        self.chart_init()
        self.timer_init()

        # 播放时长, 以 input 的时长为准
        self.video_length = 0

        # 推理使用另外一线程
        self.predict_handler_thread = PredictHandlerThread(self.output_player,
                                                           self.predict_info_plainTextEdit,
                                                           self.predict_progressBar,
                                                           self.fps_label)

    def timer_init(self):
        # 使用QTimer，2秒触发一次，更新数据
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.drawLine)
        self.timer.start(2000)

    def chart_init(self):
        # self.gpu_info_chart._chart = QChart(title="折线图堆叠")  # 创建折线视图
        self.gpu_info_chart._chart = QChart()  # 创建折线视图
        # chart._chart.setBackgroundVisible(visible=False)      # 背景色透明
        self.gpu_info_chart._chart.setBackgroundBrush(QBrush(QColor("#FFFFFF")))  # 改变图背景色

        self.series = QSplineSeries()
        # 设置曲线名称
        self.series.setName("实时数据")
        # 把曲线添加到QChart的实例中
        self.gpu_info_chart._chart.addSeries(self.series)
        # 声明并初始化X轴，Y轴
        self.dtaxisX = QDateTimeAxis()
        self.vlaxisY = QValueAxis()
        # 设置坐标轴显示范围
        self.dtaxisX.setMin(QDateTime.currentDateTime().addSecs(-300 * 1))
        self.dtaxisX.setMax(QDateTime.currentDateTime().addSecs(0))
        self.vlaxisY.setMin(0)
        self.vlaxisY.setMax(1500)
        # 设置X轴时间样式
        self.dtaxisX.setFormat("MM月dd hh:mm:ss")
        # 设置坐标轴上的格点
        self.dtaxisX.setTickCount(6)
        self.vlaxisY.setTickCount(11)
        # 设置坐标轴名称
        self.dtaxisX.setTitleText("时间")
        self.vlaxisY.setTitleText("量程")
        # 设置网格不显示
        self.vlaxisY.setGridLineVisible(False)
        # 把坐标轴添加到chart中
        self.gpu_info_chart._chart.addAxis(self.dtaxisX, Qt.AlignBottom)
        self.gpu_info_chart._chart.addAxis(self.vlaxisY, Qt.AlignLeft)
        # 把曲线关联到坐标轴
        self.series.attachAxis(self.dtaxisX)
        self.series.attachAxis(self.vlaxisY)

        self.gpu_info_chart.setChart(self.gpu_info_chart._chart)

    def drawLine(self):
        # 获取当前时间
        bjtime = QDateTime.currentDateTime()
        # 更新X轴坐标
        self.dtaxisX.setMin(QDateTime.currentDateTime().addSecs(-300 * 1))
        self.dtaxisX.setMax(QDateTime.currentDateTime().addSecs(0))
        # 当曲线上的点超出X轴的范围时，移除最早的点
        if (self.series.count() > 149):
            self.series.removePoints(0, self.series.count() - 149)
        # 产生随即数
        yint = random.randint(0, 1500)
        # 添加数据到曲线末端
        self.series.append(bjtime.toMSecsSinceEpoch(), yint)

    # def create_series(self, dataTable, chart):
    #     # 创建折线的函数
    #     for i, data_list in enumerate(dataTable):  # [index,[list]]
    #         # 创建曲线
    #         series = QLineSeries(chart._chart)
    #         # 设置折线名
    #         series.setName(data_list[0])
    #
    #         for j, v in enumerate(data_list[1:]):
    #             #  添加折线和对应的坐标点
    #             series.append(j, v)
    #
    #         series.setPointsVisible(True)  # 显示原点
    #         # 鼠标悬停连接事件
    #         series.hovered.connect(self.onSeriesHoverd)
    #         chart._chart.addSeries(series)  # 添加折线到视图窗口
    #     return chart._chart
    #
    # def customAxisX(self, chart):
    #     # 自定义x轴(均分)
    #     chart = chart
    #     series = chart.series()
    #     if not series:
    #         return
    #     # 获取当前时间前8小时的一小时内的时间
    #     time = []
    #     for index in range(13):
    #         num = 60 / 13
    #         last_day = (datetime.datetime.now() + datetime.timedelta(hours=-8, minutes=- index * num)).strftime(
    #             "%H:%M")
    #         time.append(last_day)
    #     category = list(reversed(time))
    #
    #     '''QValueAxis是轴的范围什么的不需要自己指定，轴上显示的label（也就是0,1,2,3这些内容）是默认的。
    #     qt会根据你轴上的点自动设置。若你需要自定义一些内容，QCategoryAxis是比较好的，但是需要自己自定义好才可以调用。'''
    #     axisx = QCategoryAxis(
    #         chart, labelsPosition=QCategoryAxis.AxisLabelsPositionOnValue)
    #
    #     axisx.setGridLineVisible(False)  # 隐藏网格线条
    #     axisx.setTickCount(len(category))  # 设置刻度个数
    #     minx = chart.axisX().min()
    #     maxx = chart.axisX().max()
    #     tickc = chart.axisX().tickCount()
    #     print(tickc)
    #     if tickc < 2:
    #         axisx.append(category[0])
    #     else:
    #         step = (maxx - minx) / (tickc - 1)  # tickc>=2
    #         for i in range(0, tickc):
    #             axisx.append(category[i], minx + i * step)
    #             # 保存x轴值
    #     chart.setAxisX(axisx, series[-1])
    #
    # def onSeriesHoverd(self, point, state):
    #     # 鼠标悬停事件(底部x,y)
    #     if state:
    #         try:
    #             name = self.sender().name()
    #         except:
    #             # QCursor.pos()悬停提示文字显示的位置
    #             name = ""
    #         QToolTip.showText(QCursor.pos(), "%s\nx: %s\ny: %s" %
    #                           (name, point.x(), point.y()))

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
        self.predict_handler_thread.parameter_source = self.parameter_source.toLocalFile()
        self.input_player.pause()  # 显示媒体
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
