import os
import sys
import time
import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QIcon, QImage, QPainter, QPixmap
from PyQt5.QtWidgets import (QApplication, QGridLayout, QHBoxLayout, QLabel,
                             QMessageBox, QPushButton, QVBoxLayout, QWidget)

import cv2 as cv
from model_build import (TRASH_DICT, TRASH_KIND, build_general_model,
                         build_prune_model)

record_path = '../image_record'


def get_trash_kind(num):
    trash_kind = TRASH_DICT.get(num)
    trash_class = TRASH_KIND.get(trash_kind)
    return trash_class


def cv_image2Qt(image, _h, _w):
    img = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    img = cv.resize(img, (_h, _w), interpolation=cv.INTER_LINEAR)
    height, width, bytesPerComponent = img.shape
    bytesPerLine = 3 * width
    QImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
    return QImg


class Example2(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()
        self.timer_camera = QTimer()
        self.cap = cv.VideoCapture()
        self.CAM_NUM = 1
        self.slot_init()
        self.__flag_work = 0
        self.model = build_general_model()
        self.img_class = []

    def initUI(self):
        grid = QGridLayout()
        self.setLayout(grid)
        self.data_show_layout = QHBoxLayout(self)
        self.camera_toge_layout = QVBoxLayout(self)
        self.result_toge_layout = QVBoxLayout(self)
        self.qrcode_toge_layout = QVBoxLayout(self)
        self.button_toge_layout = QVBoxLayout(self)

        self.Button_open_camera = QPushButton()
        self.Button_to_exit = QPushButton()
        self.Button_to_save = QPushButton()
        self.Button_to_predict = QPushButton()

        self.height_for_iamge = 500
        self.width_for_image = 500
        self.label_show_camera = QLabel(self)
        self.label_show_camera.setFixedSize(
            self.height_for_iamge, self.width_for_image)
        self.label_show_camera.setContentsMargins(10, 10, 10, 10)
        self.label_show_camera.setStyleSheet(
            "border:2px groove gray;border-radius:10px;padding:2px 4px;")
        self.label_show_result = QLabel(self)
        self.label_show_result.setFixedSize(
            self.height_for_iamge, self.width_for_image)
        self.label_show_result.setContentsMargins(10, 10, 10, 10)
        self.label_show_result.setStyleSheet(
            "border:2px groove gray;border-radius:10px;padding:2px 4px;")

        self.label_show_qrcode = QLabel(self)
        self.label_show_qrcode.setFixedSize(
            self.height_for_iamge, self.width_for_image)
        self.label_show_qrcode.setContentsMargins(10, 10, 10, 10)
        self.label_show_qrcode.setStyleSheet(
            "border:2px groove gray;border-radius:10px;padding:2px 4px;")

        self.label_show_camera.setScaledContents(True)
        self.label_show_result.setScaledContents(True)
        self.label_show_qrcode.setScaledContents(True)

        self.Button_open_camera.setText('open camera')
        self.Button_open_camera.setFont(QFont("Timers", 14, QFont.Bold))
        self.Button_open_camera.setStyleSheet("QPushButton{color:black}"
                                              "QPushButton:hover{color:red}"
                                              "QPushButton{background-color:lightgreen}"
                                              "QPushButton{border:2px}"
                                              "QPushButton{border-radius:10px}"
                                              "QPushButton{padding:2px 4px}")

        self.Button_to_exit.setText('exit')
        self.Button_to_exit.setFont(QFont("Timers", 14, QFont.Bold))
        self.Button_to_exit.setStyleSheet("QPushButton{color:black}"
                                          "QPushButton:hover{color:red}"
                                          "QPushButton{background-color:lightgreen}"
                                          "QPushButton{border:2px}"
                                          "QPushButton{border-radius:10px}"
                                          "QPushButton{padding:2px 4px}")

        self.Button_to_save.setText('save')
        self.Button_to_save.setFont(QFont("Timers", 14, QFont.Bold))
        self.Button_to_save.setStyleSheet("QPushButton{color:black}"
                                          "QPushButton:hover{color:red}"
                                          "QPushButton{background-color:lightgreen}"
                                          "QPushButton{border:2px}"
                                          "QPushButton{border-radius:10px}"
                                          "QPushButton{padding:2px 4px}")

        self.Button_to_predict.setText('predict')
        self.Button_to_predict.setFont(QFont("Timers", 14, QFont.Bold))
        self.Button_to_predict.setStyleSheet("QPushButton{color:black}"
                                             "QPushButton:hover{color:red}"
                                             "QPushButton{background-color:lightgreen}"
                                             "QPushButton{border:2px}"
                                             "QPushButton{border-radius:10px}"
                                             "QPushButton{padding:2px 4px}")

        self.Button_open_camera.setFixedSize(200, 50)
        self.Button_to_exit.setFixedSize(150, 50)
        self.Button_to_save.setFixedSize(150, 50)
        self.Button_to_predict.setFixedSize(150, 50)

        # default image to show
        default_image1 = cv.imread('background0.jpg')
        self.pixmap1 = QPixmap.fromImage(cv_image2Qt(
            default_image1, self.height_for_iamge, self.width_for_image))
        default_image2 = cv.imread('background4.jpg')
        self.pixmap2 = QPixmap(cv_image2Qt(
            default_image2, self.height_for_iamge, self.width_for_image))
        default_image3 = cv.imread('QRcode1.png')
        self.pixmap3 = QPixmap(cv_image2Qt(
            default_image3, self.height_for_iamge, self.width_for_image))

        self.label_show_camera.setPixmap(self.pixmap1)
        self.label_show_result.setPixmap(self.pixmap2)
        #self.label_show_qrcode.setPixmap(self.pixmap3)

        self.N1 = QLabel(self)
        self.N1.setFixedSize(self.width_for_image, 50)
        self.N1.setText('<b>real_time_image</b>')
        self.N1.setFont(QFont("微软雅黑", 14, QFont.Bold))
        self.N1.setAlignment(Qt.AlignCenter)
        self.N1.setStyleSheet('background-color: rgb(255, 251, 100)')
        self.N2 = QLabel(self)
        self.N2.setFixedSize(self.width_for_image, 50)
        self.N2.setText('<b>prediction</b>')
        self.N2.setFont(QFont("微软雅黑", 14, QFont.Bold))
        self.N2.setAlignment(Qt.AlignCenter)
        self.N2.setStyleSheet('background-color: rgb(255, 251, 100)')
        self.N3 = QLabel(self)
        self.N3.setFixedSize(self.width_for_image, 50)
        self.N3.setText('<b>QR_code</b>')
        self.N3.setFont(QFont("微软雅黑", 14, QFont.Bold))
        self.N3.setAlignment(Qt.AlignCenter)
        self.N3.setStyleSheet('background-color: rgb(255, 251, 100)')

        # grid.addWidget(self.Button_open_camera, 2, 22, alignment=Qt.AlignCenter)
        # grid.addWidget(self.Button_to_exit, 4, 22, alignment=Qt.AlignCenter)
        # grid.addWidget(self.N1, 5, 2)
        # grid.addWidget(self.N2, 5, 8)
        # grid.addWidget(self.N3, 5, 15)
        # self.data_show_layout.addWidget(self.label_show_camera)
        # self.data_show_layout.addWidget(self.label_show_result)
        # self.data_show_layout.addWidget(self.label_show_qrcode)

        self.camera_toge_layout.addWidget(self.label_show_camera)
        self.camera_toge_layout.addWidget(self.N1)
        self.result_toge_layout.addWidget(self.label_show_result)
        self.result_toge_layout.addWidget(self.N2)
        self.qrcode_toge_layout.addWidget(self.label_show_qrcode)
        self.qrcode_toge_layout.addWidget(self.N3)
        self.button_toge_layout.addWidget(
            self.Button_open_camera, alignment=Qt.AlignCenter)
        self.button_toge_layout.addWidget(
            self.Button_to_predict, alignment=Qt.AlignCenter)
        self.button_toge_layout.addWidget(
            self.Button_to_save, alignment=Qt.AlignCenter)
        self.button_toge_layout.addWidget(
            self.Button_to_exit, alignment=Qt.AlignCenter)

        self.data_show_layout.addLayout(self.camera_toge_layout)
        self.data_show_layout.addLayout(self.result_toge_layout)
        self.data_show_layout.addLayout(self.qrcode_toge_layout)
        self.data_show_layout.addLayout(self.button_toge_layout)

        grid.addLayout(self.data_show_layout, 1, 0, 5,
                       20, alignment=Qt.AlignCenter)

        self.move(300, 200)
        # self.setFixedSize(1400, 500)
        self.setWindowIcon(QIcon('Icon.jpg'))
        self.setWindowTitle('Testing_PyQt_UI')

    def paintEvent(self, event):
        painter = QPainter(self)
        image = cv.imread('background3.jpg')
        pixmap = QPixmap.fromImage(cv_image2Qt(image, 1400, 500))
        painter.drawPixmap(self.rect(), pixmap)

    def slot_init(self):
        self.Button_open_camera.clicked.connect(self.button_open_camera_click)
        self.Button_to_exit.clicked.connect(self.close)
        self.Button_to_save.clicked.connect(self.save_image)
        self.timer_camera.timeout.connect(self.show_camera)
        self.Button_to_predict.clicked.connect(self.predict)

    def button_open_camera_click(self):
        if self.timer_camera.isActive() == False:
            flag = self.cap.open(self.CAM_NUM)
            if flag == False:
                msg = QMessageBox.warning(self, u"Warning",
                                          u"Please check out the connection between camera and PC is correct !",
                                          buttons=QMessageBox.Ok,
                                          defaultButton=QMessageBox.Ok)

            else:
                # Starts or restarts the timer with a timeout of duration msec milliseconds.
                self.timer_camera.start(30)
                self.Button_open_camera.setText(u'close camera')
        else:
            self.timer_camera.stop()
            self.cap.release()
            self.label_show_camera.clear()
            self.label_show_result.clear()
            self.label_show_qrcode.clear()
            self.label_show_camera.setPixmap(self.pixmap1)
            self.label_show_result.setPixmap(self.pixmap2)
            self.Button_open_camera.setText(u'open camera')
            self.N2.setText("prediction")

    def show_camera(self):
        flag, self.image = self.cap.read()
        showImage = cv_image2Qt(
            self.image, self.height_for_iamge, self.width_for_image)
        self.label_show_camera.setPixmap(QPixmap.fromImage(showImage))

    def predict(self):

        if self.timer_camera.isActive() == False:
            button = QMessageBox.warning(self, "Warning", "Please open the camera first! ", QMessageBox.Ok,
                                         QMessageBox.Ok)
            if button == QMessageBox.Ok:
                pass
        else:
            flag, image = self.cap.read()
            self.timer_camera.stop()
            start = time.process_time()
            img_to_test = cv.resize(
                image, (224, 224), interpolation=cv.INTER_LINEAR)
            image_to_test = np.reshape(img_to_test, (1, 224, 224, 3))
            prediction = self.model.predict(image_to_test)
            pred_label = np.argsort(prediction[0])[::-1][0:2]
            first_pred_pro = sorted(prediction[0], reverse=True)[0]
            first_pred_kind = get_trash_kind(str(pred_label[0]))
            end = time.process_time()

            if first_pred_kind == 'Recyclable Waste':
                self.img_class = cv.imread('Recyclable.jpg')
            elif first_pred_kind == 'Residual Waste':
                self.img_class = cv.imread('Residual.jpg')
            elif first_pred_kind == 'Household Waste':
                self.img_class = cv.imread('Household.jpg')
            else:
                self.img_class = cv.imread('Hazardous.jpg')
            showimage = cv_image2Qt(
                self.img_class, self.height_for_iamge, self.width_for_image)
            self.label_show_result.setPixmap(QPixmap.fromImage(showimage))
            QMessageBox.information(self, "Information",
                                    first_pred_kind + " with probability: " + str(first_pred_pro) + "\n" +
                                    "Time consumption: " + str(end - start))
            self.N2.setText("Last prediction: "+str(first_pred_kind))
            self.label_show_qrcode.setPixmap(self.pixmap3)
            self.timer_camera.start(30)

    def save_image(self):
        if self.timer_camera.isActive() == False:
            button = QMessageBox.warning(self, "Warning", "Please open the camera first! ", QMessageBox.Ok,
                                         QMessageBox.Ok)
            if button == QMessageBox.Ok:
                pass
        else:
            flag, image = self.cap.read()
            self.timer_camera.stop()
            image_tested = len(os.listdir(record_path))
            datetime = str(time.strftime(
                '%Y%m%d%H%M%S', time.localtime(time.time())))
            img_to_save = cv.resize(
                image, (224, 224), interpolation=cv.INTER_LINEAR)

            button = QMessageBox.question(self, "Question", "Save image ? ",
                                          QMessageBox.Ok | QMessageBox.Cancel, QMessageBox.Ok)

            if button == QMessageBox.Ok:
                path = os.path.join(record_path, datetime +
                                    '-' + str(image_tested + 1) + '.jpg')
                cv.imwrite(path, img_to_save)
                if os.path.exists(path):
                    QMessageBox.information(
                        self, "Information", "Save successfully")
                else:
                    QMessageBox.information(
                        self, "Information", "Save failed, please check your code!")
            elif button == QMessageBox.Cancel:
                pass
            else:
                pass

            self.timer_camera.start(30)

    def closeEvent(self, event):
        ok = QPushButton()
        cancel = QPushButton()

        msg = QMessageBox(QMessageBox.Warning, u"Exit", u"Sure to exit ?")

        msg.addButton(ok, QMessageBox.ActionRole)
        msg.addButton(cancel, QMessageBox.RejectRole)
        ok.setText(u'Yes')
        cancel.setText(u'No')

        if msg.exec_() == QMessageBox.RejectRole:
            event.ignore()
        else:
            #             self.socket_client.send_command(self.socket_client.current_user_command)
            if self.cap.isOpened():
                self.cap.release()
            if self.timer_camera.isActive():
                self.timer_camera.stop()
            event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Example2()
    ex.show()
    sys.exit(app.exec_())
