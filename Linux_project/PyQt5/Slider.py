from PyQt5.QtWidgets import QWidget, QSlider, QLabel, QApplication,QMessageBox
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
import sys


class Example(QWidget):

    def __init__(self):
        super().__init__()

        self.initUI()

    def changeValue(self, value):
        if value == 0:
            self.label.setPixmap(QPixmap('Harmful.jpg'))
        elif value >= 0 and value <= 30:
            self.label.setPixmap(QPixmap('Recyclable.jpg'))
        elif value > 30 and value < 80:
            self.label.setPixmap(QPixmap('Residual.jpg'))
        else:
            self.label.setPixmap(QPixmap('Household.jpg'))

    def initUI(self):

        sld = QSlider(Qt.Horizontal, self)
        sld.setFocusPolicy(Qt.NoFocus)
        sld.setGeometry(30, 40, 100, 30)
        sld.valueChanged[int].connect(self.changeValue)

        self.label = QLabel(self)
        self.label.setPixmap(QPixmap('Recyclable.jpg'))
        self.label.setGeometry(160,50, 250, 250)

        self.setGeometry(300, 300, 500, 500)
        self.setWindowTitle('QSlider')
        self.show()

    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Message',
                                     "Are you sure to quit?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())
