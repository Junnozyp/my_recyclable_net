from PyQt5.QtWidgets import (QWidget, QPushButton,
                             QFrame, QApplication)
from PyQt5.QtGui import QColor
import sys


class Example(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        # cb = QCheckBox('generate QR code relevant to the result', self)
        #
        # cb.move(20, 20)
        # cb.toggle()
        # cb.stateChanged.connect(self.changeTitle)

        self.col = QColor(0, 0, 0)
        redbnt = QPushButton('Red', self)
        redbnt.setCheckable(True)
        redbnt.move(10, 10)
        redbnt.clicked[bool].connect(self.setColor)

        greenbnt = QPushButton('Green', self)
        greenbnt.setCheckable(True)
        greenbnt.move(10, 60)
        greenbnt.clicked[bool].connect(self.setColor)

        bluebnt = QPushButton('Bule', self)
        bluebnt.setCheckable(True)
        bluebnt.move(10, 110)
        bluebnt.clicked[bool].connect(self.setColor)
        # creat a square zone
        self.square = QFrame(self)
        self.square.setGeometry(150, 20, 100, 100)
        self.square.setStyleSheet("QWidget { background-color: %s }" % self.col.name())

        self.setGeometry(300, 300, 280, 170)
        self.setWindowTitle('Toggle button')
        self.show()
    def setColor(self,pressed):
        source =self.sender()

        if pressed:
            val=255
        else:
            val=0
        if source.text() == "Red":
            self.col.setRed(val)
        elif source.text() == "Green":
            self.col.setGreen(val)
        else:
            self.col.setBlue(val)

        self.square.setStyleSheet("QFrame { background-color: %s }" %
            self.col.name())

            # def changeTitle(self, state):
    #     if state == Qt.Checked:
    #         self.setWindowTitle('QCheckbox')
    #     else:
    #         self.setWindowTitle(' ')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())
