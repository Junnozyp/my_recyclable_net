#!/usr/bin/python
# -*- coding: UTF-8 -*-
import sys
import cv2

from PyQt5 import QtCore, QtGui, QtWidgets
import os


# reload(sys)
# sys.setdefaultencoding('utf-8')

class Ui_MainWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(Ui_MainWindow, self).__init__(parent)

        # 定时器
        self.timer_camera = QtCore.QTimer()
        self.cap = cv2.VideoCapture()
        self.CAM_NUM = 1
        self.set_ui()
        self.slot_init()
        self.__flag_work = 0
        self.x = 0

    def set_ui(self):

        self.__layout_main = QtWidgets.QHBoxLayout()
        self.__layout_fun_button = QtWidgets.QHBoxLayout()
        self.__layout_fun_button_V = QtWidgets.QVBoxLayout()
        self.__layout_data_show = QtWidgets.QVBoxLayout()

        # 按键设置
        self.button_open_camera = QtWidgets.QPushButton(u'打开相机')
        self.button_exit = QtWidgets.QPushButton(u'退出')
        self.button_open_camera.setMinimumHeight(50)
        self.button_exit.setMinimumHeight(50)
        self.button_exit.move(10, 100)

        # 信息显示
        self.label_show_camera = QtWidgets.QLabel()

        # self.label_move = QtWidgets.QLabel()
        # self.label_move.setFixedSize(200, 200)

        
        self.label_show_camera.setGeometry(0, 0, 225, 225)
        # self.label_show_camera.setAutoFillBackground(False)
        #self.label_show_camera.setScaledContents(True)

        self.__layout_fun_button.addWidget(self.button_open_camera)
        self.__layout_fun_button.addWidget(self.button_exit)
        self.__layout_fun_button.addStretch(1)
        self.__layout_fun_button_V.addStretch(1)
        self.__layout_fun_button_V.addWidget(self.label_show_camera)
        self.__layout_fun_button_V.addLayout(self.__layout_fun_button)

        # self.__layout_fun_button.addWidget(self.label_move)

        self.__layout_main.addLayout(self.__layout_fun_button_V)
        #self.__layout_main.addWidget(self.label_show_camera)

        self.setLayout(self.__layout_main)
        self.setGeometry(300, 300, 500, 500)
        # self.label_move.raise_()
        self.setWindowTitle(u'real_time_image_showing')

    # def mousePressEvent(self, QMouseEvent):
    #     x = QMouseEvent.x()
    #     y = QMouseEvent.y()
    #     self.label_move.move(0,0)
    #     print(x,y)
    #     print(self.label_move.pos())

    def slot_init(self):
        self.button_open_camera.clicked.connect(self.button_open_camera_click)
        self.timer_camera.timeout.connect(self.show_camera)
        self.button_exit.clicked.connect(self.close)

    # 未开启时Timer没有运行，即self.timer_camera.isActive() == False，
    def button_open_camera_click(self):
        if self.timer_camera.isActive() == False:
            flag = self.cap.open(self.CAM_NUM)
            if flag == False:
                msg = QtWidgets.QMessageBox.warning(self, u"Warning", u"请检测相机与电脑是否连接正确",
                                                    buttons=QtWidgets.QMessageBox.Ok,
                                                    defaultButton=QtWidgets.QMessageBox.Ok)
            # if msg==QtGui.QMessageBox.Cancel:
            #                     pass
            else:
                # Starts or restarts the timer with a timeout of duration msec milliseconds.
                self.timer_camera.start(30)

                self.button_open_camera.setText(u'关闭相机')
        else:
            self.timer_camera.stop()
            self.cap.release()
            self.label_show_camera.clear()
            self.button_open_camera.setText(u'打开相机')

    def show_camera(self):
        flag, self.image = self.cap.read()
        # face = self.face_detect.align(self.image)
        # if face:
        #     pass
        show = cv2.resize(self.image, (224, 224), interpolation=cv2.INTER_LINEAR)
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
        self.label_show_camera.setPixmap(QtGui.QPixmap.fromImage(showImage))
        # self.x += 1
        # self.label_move.move(self.x,100)

        # if self.x ==320:
        #     self.label_show_camera.raise_()

    def closeEvent(self, event):
        ok = QtWidgets.QPushButton()
        cancel = QtWidgets.QPushButton()

        msg = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning, u"关闭", u"是否关闭！")

        msg.addButton(ok, QtWidgets.QMessageBox.ActionRole)
        msg.addButton(cancel, QtWidgets.QMessageBox.RejectRole)
        ok.setText(u'确定')
        cancel.setText(u'取消')
        # msg.setDetailedText('sdfsdff')
        if msg.exec_() == QtWidgets.QMessageBox.RejectRole:
            event.ignore()
        else:
            #             self.socket_client.send_command(self.socket_client.current_user_command)
            if self.cap.isOpened():
                self.cap.release()
            if self.timer_camera.isActive():
                self.timer_camera.stop()
            event.accept()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ui = Ui_MainWindow()
    ui.show()
    sys.exit(app.exec_())
