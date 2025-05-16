from PyQt6.QtCore import *
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6 import *
from ultralytics import YOLO
from collections import Counter
from pathlib import Path
import sys
import os

dir_path = os.path.dirname(os.path.abspath(__file__))

class MainWidow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.file_selected = False
        self.model = YOLO(dir_path + r"\models\yolov8n.pt")
        self.setFixedSize(QSize(1500, 700))
        self.setWindowTitle("CASCAD EYE")
        self.setWindowIcon(QIcon(QPixmap(dir_path + r"\icons\icon.png")))
        
        self.toolbar = QToolBar(self)
        self.search_btn = QPushButton("Начать обнаружение", self)
        self.search_btn.setFixedSize(QSize(150, 40))
        self.search_btn.move(QPoint(1330, 640))
        self.search_btn.clicked.connect(self.detect_objects)

        self.open_file_btn = QPushButton("Найти изображение", self)
        self.open_file_btn.setFixedSize(QSize(150, 40))
        self.open_file_btn.move(QPoint(1160, 640))      
        self.open_file_btn.clicked.connect(self.getfile)

        self.clear_btn = QPushButton("Сбросить изображение", self)
        self.clear_btn.move(QPoint(20, 640))
        self.clear_btn.setFixedSize(QSize(150, 40))
        self.clear_btn.clicked.connect(self.clear)

        self.model_select_box = QComboBox(self)
        self.model_select_box.setFixedSize(QSize(150, 40))
        self.model_select_box.move(QPoint(990, 640))  
        self.models = ["YOLO v8N", "YOLO v11N", "YOLO v11X"]
        self.model_select_box.activated.connect(self.model_select) 
        for model in self.models:
           self.model_select_box.addItem(model)

        self.le = QLabel(parent=self)
        self.le.move(QPoint(400, 100))

        self.label = QLabel(parent=self, text="Здесь будет ваше изображение")
        self.label.move(QPoint(650, 350))
        self.label.setFixedSize(QSize(200, 20))

        self.wait_label = QLabel("Пожалуйста, подождите...", self)
        self.wait_label.setVisible(False)
        self.wait_label.setFixedSize(QSize(150, 40))
        self.wait_label.move(QPoint(820, 640))


    def getfile(self):
      self.fname = QFileDialog.getOpenFileName(self, 'Открыть файл', None, "Изображения (*.png, *.jpg)")
      if self.fname[0] != None:
         self.file_selected = True
      else:
         self.file_selected = False

      self.pixmap = QPixmap(self.fname[0])
      self.le.setPixmap(self.pixmap)
      self.le.setFixedSize(QSize(700, 500))
      self.label.setVisible(False)
      self.le.setVisible(True)
    
    def clear(self):
       self.file_selected = False
       self.label.setVisible(True)
       self.le.setVisible(False)

    def model_select(self, idx):
        self.model_selected = self.models[idx]
        if self.model_selected == "YOLO v8N":
           self.model = YOLO(dir_path + r"\models\yolov8n.pt")
        elif self.model_selected == "YOLO v11N": 
           self.model = YOLO(dir_path + r"\models\yolo11n.pt")
        elif self.model_selected == "YOLO v11X":   
           self.model = YOLO(dir_path + r"\models\yolo11x.pt")
        else:
           pass

    def detect_objects(self) -> None:
        if self.file_selected == True:
           self.wait_label.setVisible(True)
           self.results = self.model(self.fname[0], verbose=False)[0]
           self.save_path = self.results.save(filename=f"RESULT_{Path(self.fname[0]).stem}.png")
           self.le.setPixmap(QPixmap(f"RESULT_{Path(self.fname[0]).stem}.png"))
           self.msgbox = QMessageBox(self)
           self.msgbox.setIcon(QMessageBox.Icon.Information)
           self.msgbox.setText(f"Результат сохранён в файл: {self.save_path}")
           self.msgbox.setWindowTitle("Информация")
           self.msgbox.show()
           self.wait_label.setVisible(False)
        else:
           self.error_msgbox = QMessageBox(self)
           self.error_msgbox.setIcon(QMessageBox.Icon.Critical)
           self.error_msgbox.setText("Файл не выбран!")
           self.error_msgbox.setWindowTitle("Ошибка")
           self.error_msgbox.show()
            
app = QApplication(sys.argv)
app.setStyle("Fusion")
window = MainWidow()
window.show()
app.exec()      


        
        