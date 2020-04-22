import sys
import os
import cv2
import webbrowser
import numpy as np
import mxnet as mx
import time
import matplotlib.pyplot as plt
from PyQt5 import QtCore,QtWidgets,QtGui
from PyQt5.QtWidgets import QFileDialog,QApplication,QMessageBox
from gluoncv import model_zoo, data, utils
from PIL import Image
from get_color import get_dominant_color,get_color
os.environ['MXNET_GLUON_REPO']='https://apache-mxnet.s3.cn-north-1.amazonaws.com.cn/'


class Ui_MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui_MainWindow, self).__init__()
        self.setupUi(self)
        self.retranslateUi(self)
        self.CLASSES=['blouse', 'blazer', 'tee', 'tank', 'top', 'sweater', 'hoodie', 'cardigan', 'jacket', 'skirt', 'shorts', 'jeans', 'joggers', 'sweatpants', 'cutoffs', 'sweatshorts', 'leggings', 'dress', 'romper', 'coat', 'kimono', 'jumpsuit']
        self.CLASSES_2_TYPE={'anorak': 'upper-body clothes', 'blazer': 'upper-body clothes', 'blouse': 'upper-body clothes', 'bomber': 'upper-body clothes', 'button-down': 'upper-body clothes', 'cardigan': 'upper-body clothes', 'flannel': 'upper-body clothes', 'halter': 'upper-body clothes', 'henley': 'upper-body clothes', 'hoodie': 'upper-body clothes', 'jacket': 'upper-body clothes', 'jersey': 'upper-body clothes', 'parka': 'upper-body clothes', 'peacoat': 'upper-body clothes', 'poncho': 'upper-body clothes', 'sweater': 'upper-body clothes', 'tank': 'upper-body clothes', 'tee': 'upper-body clothes', 'top': 'upper-body clothes', 'turtleneck': 'upper-body clothes', 'capris': 'lower-body clothes', 'chinos': 'lower-body clothes', 'culottes': 'lower-body clothes', 'cutoffs': 'lower-body clothes', 'gauchos': 'lower-body clothes', 'jeans': 'lower-body clothes', 'jeggings': 'lower-body clothes', 'jodhpurs': 'lower-body clothes', 'joggers': 'lower-body clothes', 'leggings': 'lower-body clothes', 'sarong': 'lower-body clothes', 'shorts': 'lower-body clothes', 'skirt': 'lower-body clothes', 'sweatpants': 'lower-body clothes', 'sweatshorts': 'lower-body clothes', 'trunks': 'lower-body clothes', 'caftan': 'full-body clothes', 'cape': 'full-body clothes', 'coat': 'full-body clothes', 'coverup': 'full-body clothes', 'dress': 'full-body clothes', 'jumpsuit': 'full-body clothes', 'kaftan': 'full-body clothes', 'kimono': 'full-body clothes', 'nightdress': 'full-body clothes', 'onesie': 'full-body clothes', 'robe': 'full-body clothes', 'romper': 'full-body clothes', 'shirtdress': 'full-body clothes', 'sundress': 'full-body clothes'}

        self.net=self.load_net()
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1096, 739)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(190, 40, 521, 31))
        self.lineEdit.setObjectName("lineEdit")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(720, 40, 111, 31))
        self.pushButton.setObjectName("pushButton")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(40, 130, 381, 511))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(460, 130, 351, 511))
        self.label_2.setObjectName("label_2")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(120, 630, 93, 28))
        self.pushButton_2.setObjectName("pushButton_2")
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar.setGeometry(QtCore.QRect(520, 210, 261, 41))
        self.progressBar.setProperty("value", 24)
        self.progressBar.setObjectName("progressBar")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(970, 330, 93, 28))
        self.pushButton_3.setObjectName("pushButton_3")
        self.lineEdit_2 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_2.setGeometry(QtCore.QRect(940, 180, 151, 21))
        self.lineEdit_2.setText("")
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.lineEdit_3 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_3.setGeometry(QtCore.QRect(940, 220, 151, 21))
        self.lineEdit_3.setText("")
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.lineEdit_4 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_4.setGeometry(QtCore.QRect(940, 260, 151, 21))
        self.lineEdit_4.setText("")
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(860, 180, 72, 15))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(860, 220, 72, 15))
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(860, 260, 72, 15))
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(860, 290, 72, 15))
        self.label_6.setObjectName("label_6")
        self.lineEdit_5 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_5.setGeometry(QtCore.QRect(940, 290, 151, 21))
        self.lineEdit_5.setText("")
        self.lineEdit_5.setObjectName("lineEdit_5")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.pushButton.clicked.connect(self.openfile)
        self.pushButton_2.setHidden(True)
        self.progressBar.setHidden(True)
        self.pushButton_2.clicked.connect(self.predict)
        self.pushButton_3.clicked.connect(self.search)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Clothes"))
        self.pushButton.setText(_translate("MainWindow", "Upload"))
        # self.label.setText(_translate("MainWindow", "origin_img"))
        # self.label_2.setText(_translate("MainWindow", "predicted_img"))
        self.pushButton_2.setText(_translate("MainWindow", "Predict"))
        self.pushButton_3.setText(_translate("MainWindow", "Search"))
        self.label_3.setText(_translate("MainWindow", "Category:"))
        self.label_4.setText(_translate("MainWindow", "Type:"))
        self.label_5.setText(_translate("MainWindow", "Color:"))
        self.label_6.setText(_translate("MainWindow", "Addtion:"))
    def openfile(self):
        self.fileName, filetype = QFileDialog.getOpenFileName(self, "Choose Files", "./",
                                                          "All Files (*);;Excel Files (*.xls)")
        if self.fileName.endswith('jpg'):
            self.lineEdit.setText(self.fileName)
            image=self.load_img(self.fileName)
            self.label.setPixmap(image)
            # self.label.setPixmap(QtGui.QPixmap(self.fileName))
            self.pushButton_2.setHidden(False)
        else:
            self.label.setText('Bad Input!Please check Input!')
    def openUrl(self,url):
        webbrowser.open_new_tab(url)
    def load_net(self):
        net = model_zoo.faster_rcnn_resnet50_v1b_voc(pretrained=True,force_nms=True)
        net.reset_class(self.CLASSES)
        net.load_parameters('./params/weight.params',allow_missing=True)
        return net

    def load_img(self,file_path):
        x, resized_img = data.transforms.presets.rcnn.load_test(file_path,short=300,max_size=480)
        # 将图片转化成Qt可读格式
        image = QtGui.QImage(resized_img,resized_img.shape[1],resized_img.shape[0], resized_img.shape[1] * 3, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap(image)
        self.x = x
        self.show_img = resized_img
        return pix

    def search(self):
        # reply=QMessageBox.question(self,"Search Engine","问答消息正文",QMessageBox.Yes|QMessageBox.No,QMessageBox.Yes)
        # if reply == QMessageBox.Yes:
        #     print('退出')
        # else:
        #     print('不退出')
        customMsgBox = QMessageBox(self)
        customMsgBox.setWindowTitle("Search Engine")
        lockButton = customMsgBox.addButton(self.tr("Amazon"),
                                            QMessageBox.ActionRole)
        unlockButton = customMsgBox.addButton(self.tr("AliExpress"),
                                              QMessageBox.ActionRole)
        cancelButton = customMsgBox.addButton("cancel", QMessageBox.ActionRole)

        customMsgBox.setText(self.tr("Please choose search engine"))
        customMsgBox.exec_()
        button = customMsgBox.clickedButton()
        category=self.lineEdit_2.text()
        category_type=self.lineEdit_3.text()
        color=self.lineEdit_4.text()
        addition=self.lineEdit_5.text()
        if addition:
            keywords=category+'+'+category_type+'+'+color+'+'+addition
        else:
            keywords = category + '+' + category_type + '+' + color
        if button == lockButton:
            self.openUrl('https://www.amazon.com/?k=%s'%keywords)
        elif button == unlockButton:
            self.openUrl('https://www.aliexpress.com/wholesale?&SearchText=%s'%keywords)

    def predict(self):
        self.progressBar.setHidden(False)
        file_path=self.fileName
        self.progressBar.setValue(10)
        # input_img=cv2.imread(file_path,0)
        x, resized_img = data.transforms.presets.rcnn.load_test(file_path)
        box_ids, scores, bboxes = self.net(x)
        self.progressBar.setValue(50)
        img,class_name,bbox_res= self.opencv_show(resized_img, bboxes[0], scores[0], box_ids[0], class_names=self.net.classes, thresh=0.5)
        if len(class_name)!=0:
            output_imgs = file_path.replace('test_imgs', 'predict')
            img = cv2.resize(img, (self.show_img.shape[1],self.show_img.shape[0]))
            time.sleep(1)
            self.progressBar.setValue(90)
            cv2.imwrite(output_imgs, img)
            time.sleep(1)
            self.progressBar.setValue(100)
            self.progressBar.setHidden(True)
            self.label_2.setPixmap(QtGui.QPixmap(output_imgs))
            self.lineEdit_2.setText(class_name[0])
            if len(class_name)==1:
                xmin, ymin, xmax, ymax=bbox_res[0]
                color_target_clothes=resized_img[ymin:ymax,xmin:xmax,:]
                color_target_clothes[:,:,(0,1,2)]=color_target_clothes[:,:,(2,1,0)]
                # color_target_clothes=Image.fromarray(color_target_clothes)
                # plt.imshow(np.asarray(color_target_clothes))
                # plt.show()
                COLOR=get_color(color_target_clothes)
                if COLOR=='red2':
                    COLOR='red'
                CLASS_TYPE=self.CLASSES_2_TYPE.get(class_name[0])
                self.lineEdit_3.setText(CLASS_TYPE)
                self.lineEdit_4.setText(COLOR)
            else:
                QMessageBox.information(self, "warning", "Multiple clothes detected!",
                                        QMessageBox.Yes)
        elif len(class_name)==0:
            self.progressBar.setHidden(True)
            QMessageBox.information(self, "warning", "No clothes detected!",
                                    QMessageBox.Yes)

    def opencv_show(self,this_img, bboxes, scores=None, labels=None, thresh=0.5, class_names=None):
        img=this_img.copy()
        img[:, :, (0, 1, 2)] = img[:, :, (2, 1, 0)]
        if len(bboxes) < 1:
            return img
        if isinstance(bboxes, mx.nd.NDArray):
            bboxes = bboxes.asnumpy()
        if isinstance(labels, mx.nd.NDArray):
            labels = labels.asnumpy()
        if isinstance(scores, mx.nd.NDArray):
            scores = scores.asnumpy()
        bbox_res=[]
        class_name_res=[]
        for i, bbox in enumerate(bboxes):
            if scores is not None and scores.flat[i] < thresh:
                continue
            if labels is not None and labels.flat[i] < 0:
                continue
            cls_id = int(labels.flat[i]) if labels is not None else -1
            bbox=[int(x) for x in bbox]
            xmin, ymin, xmax, ymax = [x if x>0 else 0 for x  in bbox]
            bbox_res.append([xmin, ymin, xmax, ymax])
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), thickness=2)
            if class_names is not None and cls_id < len(class_names):
                class_name = class_names[cls_id]
            else:
                class_name = str(cls_id) if cls_id >= 0 else ''
            class_name_res.append(class_name)
            score = '{:.3f}'.format(scores.flat[i]) if scores is not None else ''
            if class_name or score:
                cv2.putText(img, '{:s} {:s}'.format(class_name, score), (xmin, ymin-2), cv2.FONT_HERSHEY_SIMPLEX, 1.6,
                            (0, 0, 255), 3)
        self.progressBar.setValue(80)
        time.sleep(1)
        return img,class_name_res,bbox_res
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())