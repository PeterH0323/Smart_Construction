# -*- coding: utf-8 -*-
# @Time    : 2021/3/6 15:36
# @Author  : PeterH
# @Email   : peterhuang0323@outlook.com
# @File    : visual_interface.py
# @Software: PyCharm
# @Brief   :

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QApplication, QMainWindow  # QMainWindow, QApplication, QDialog, QWidget, QMessageBox
import sys

from UI.main_window import Ui_MainWindow


CODE_VER = "V0.1"


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        self.setupUi(self)
        self.setWindowTitle("VAT ROLL COMPARE LABEL TOOL" + " " + CODE_VER)
        self.showMaximized()

        # 按键绑定
        self.play_pushButton.clicked.connect(self.button_click)
        self.pause_pushButton.clicked.connect(self.button_click)

    @pyqtSlot()
    def button_click(self):
        name = self.sender().objectName()
        if name == "play_pushButton":
            print("play")
        elif name == "pause_pushButton":
            print("pause")

    @pyqtSlot()
    def closeEvent(self, *args, **kwargs):
        print("Close")


if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
