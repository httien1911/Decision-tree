import numpy as np
import pandas as pd
import seaborn as sns
import easygui
# %matplotlib inline
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.metrics import accuracy_score
from IPython.display import HTML

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMessageBox

from PyQt5.QtCore import *
pd.set_option('colheader_justify', 'center')


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1038, 622)
        font = QtGui.QFont()
        font.setFamily("MS Shell Dlg 2")
        font.setPointSize(9)
        MainWindow.setFont(font)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(30, 110, 281, 191))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.groupBox.setFont(font)
        self.groupBox.setObjectName("groupBox")
        self.xoabtn = QtWidgets.QPushButton(self.groupBox)
        self.xoabtn.setGeometry(QtCore.QRect(10, 40, 101, 23))
        self.xoabtn.setObjectName("xoabtn")
        self.roiracbtn = QtWidgets.QPushButton(self.groupBox)
        self.roiracbtn.setGeometry(QtCore.QRect(10, 90, 101, 23))
        self.roiracbtn.setObjectName("roiracbtn")
        self.thongkebtn = QtWidgets.QPushButton(self.groupBox)
        self.thongkebtn.setGeometry(QtCore.QRect(10, 140, 101, 23))
        self.thongkebtn.setObjectName("thongkebtn")
        self.input_xoa = QtWidgets.QLineEdit(self.groupBox)
        self.input_xoa.setGeometry(QtCore.QRect(140, 40, 113, 21))
        self.input_xoa.setObjectName("input_xoa")
        self.input_roirac = QtWidgets.QLineEdit(self.groupBox)
        self.input_roirac.setGeometry(QtCore.QRect(140, 90, 113, 21))
        self.input_roirac.setObjectName("input_roirac")
        self.input_thongke = QtWidgets.QLineEdit(self.groupBox)
        self.input_thongke.setGeometry(QtCore.QRect(140, 140, 113, 21))
        self.input_thongke.setObjectName("input_thongke")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(40, 30, 91, 16))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.choosebtn = QtWidgets.QPushButton(self.centralwidget)
        self.choosebtn.setGeometry(QtCore.QRect(40, 60, 71, 21))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.choosebtn.setFont(font)
        self.choosebtn.setObjectName("choosebtn")
        self.showname = QtWidgets.QLabel(self.centralwidget)
        self.showname.setGeometry(QtCore.QRect(120, 60, 51, 16))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.showname.setFont(font)
        self.showname.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.showname.setObjectName("showname")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(30, 330, 291, 211))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.groupBox_2.setFont(font)
        self.groupBox_2.setObjectName("groupBox_2")
        self.confu_matrixbtn = QtWidgets.QPushButton(self.groupBox_2)
        self.confu_matrixbtn.setGeometry(QtCore.QRect(10, 120, 111, 23))
        self.confu_matrixbtn.setObjectName("confu_matrixbtn")
        self.label_3 = QtWidgets.QLabel(self.groupBox_2)
        self.label_3.setGeometry(QtCore.QRect(10, 40, 121, 16))
        self.label_3.setObjectName("label_3")
        self.input_training = QtWidgets.QLineEdit(self.groupBox_2)
        self.input_training.setGeometry(QtCore.QRect(140, 40, 111, 21))
        self.input_training.setObjectName("input_training")
        self.draw_treebtn = QtWidgets.QPushButton(self.groupBox_2)
        self.draw_treebtn.setGeometry(QtCore.QRect(150, 120, 111, 23))
        self.draw_treebtn.setObjectName("draw_treebtn")
        self.ID3_check = QtWidgets.QCheckBox(self.groupBox_2)
        self.ID3_check.setGeometry(QtCore.QRect(40, 80, 41, 17))
        self.ID3_check.setObjectName("ID3_check")
        self.C45_check = QtWidgets.QCheckBox(self.groupBox_2)
        self.C45_check.setGeometry(QtCore.QRect(180, 80, 51, 17))
        self.C45_check.setObjectName("C45_check")
        self.cbb_save = QtWidgets.QComboBox(self.groupBox_2)
        self.cbb_save.setEnabled(True)
        self.cbb_save.setGeometry(QtCore.QRect(130, 170, 131, 22))
        self.cbb_save.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.cbb_save.setEditable(False)
        self.cbb_save.setCurrentText("")
        self.cbb_save.setMaxVisibleItems(20)
        self.cbb_save.setMaxCount(2147483646)
        self.cbb_save.setMinimumContentsLength(20)
        self.cbb_save.setObjectName("cbb_save")
        self.label_2 = QtWidgets.QLabel(self.groupBox_2)
        self.label_2.setGeometry(QtCore.QRect(10, 170, 111, 20))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.ketqua = QtWidgets.QTextBrowser(self.centralwidget)
        self.ketqua.setGeometry(QtCore.QRect(340, 30, 681, 551))
        self.ketqua.setObjectName("ketqua")
        self.ketqua.setLineWrapMode(QtWidgets.QTextBrowser.NoWrap)
        self.ketqua.horizontalScrollBar().setValue(0)

        self.clearbtn = QtWidgets.QPushButton(self.centralwidget)
        self.clearbtn.setGeometry(QtCore.QRect(120, 550, 81, 23))
        self.clearbtn.setObjectName("clearbtn")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1038, 21))
        self.menubar.setObjectName("menubar")
        self.menuB_i_to_n_ph_n_l_p_d_li_u = QtWidgets.QMenu(self.menubar)
        self.menuB_i_to_n_ph_n_l_p_d_li_u.setObjectName("menuB_i_to_n_ph_n_l_p_d_li_u")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar.addAction(self.menuB_i_to_n_ph_n_l_p_d_li_u.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.df = None
        self.roiracbtn.clicked.connect(self.roirac)
        # click de chon file
        self.choosebtn.clicked.connect(self.openFile)
        self.xoabtn.clicked.connect(self.xoa)
        self.confu_matrixbtn.clicked.connect(self.cal_matrix)
        self.draw_treebtn.clicked.connect(self.draw)
        self.thongkebtn.clicked.connect(self.thong_ke)
        self.clearbtn.clicked.connect(self.clear)
        self.cbb_save.currentIndexChanged.connect(self.index_changed_callback)
        self.ID3_check.stateChanged.connect(self.uncheck)
        self.C45_check.stateChanged.connect(self.uncheck)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Phân tích nhu cầu khách hàng"))
        self.groupBox.setTitle(_translate("MainWindow", "Tiền xử lý dữ liệu"))
        self.xoabtn.setText(_translate("MainWindow", "Xóa thuộc tính"))
        self.roiracbtn.setText(_translate("MainWindow", "Rời rạc hóa"))
        self.thongkebtn.setText(_translate("MainWindow", "Thống kê"))
        self.label.setText(_translate("MainWindow", "Chọn file .csv"))
        self.choosebtn.setText(_translate("MainWindow", "Chọn"))
        self.showname.setText(_translate("MainWindow", ".csv"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Phân lớp dữ liệu"))
        self.confu_matrixbtn.setText(_translate("MainWindow", "Ma trận nhầm lẫn"))
        self.label_3.setText(_translate("MainWindow", "Tỷ lệ dữ liệu training"))
        self.draw_treebtn.setText(_translate("MainWindow", "Cây quyết định"))
        self.ID3_check.setText(_translate("MainWindow", "ID3"))
        self.C45_check.setText(_translate("MainWindow", "CART"))
        self.label_2.setText(_translate("MainWindow", "Danh sách kết quả"))
        self.ketqua.setHtml(_translate("MainWindow",
                                       "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
                                       "<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
                                       "p, li { white-space: pre-wrap; }\n"
                                       "</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
                                       "<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>"))
        self.clearbtn.setText(_translate("MainWindow", "Restart"))
        self.menuB_i_to_n_ph_n_l_p_d_li_u.setTitle(_translate("MainWindow", "Bài toán phân lớp dữ liệu"))

    def openFile(self):
        self.path = easygui.fileopenbox()
        with open(self.path) as f:
            self.df = pd.read_csv(f, dtype={'age': np.int64, 'job': np.object_, 'education': np.object_,
                                            'default': np.object_, 'balance': np.int64, 'loan': np.object_,
                                            'contact': np.object_, 'day': np.int64, 'month': np.object_,
                                            'duration': np.float64, 'campaign': np.int64, 'pdays': np.int64,
                                            'previous': np.int64, 'poutcome': np.object_, 'y': np.object_})
        print(self.df.head(12))
        filename = self.path.split('''\\''')
        self.showname.setText(filename[-1])
        describe = self.df.describe()
        table = self.df.head(50)
        # show = pd.concat([table, describe])
        show = table.to_html(border=0)
        print(show)
        result = show.replace('<table border="0" class="dataframe">',
                              '<table style="text-align: center; font-size: 11pt; font-family: Arial; ">')
        result = result.replace('<td>', '<td style=" border: 1px solid silver;text-align: center;">')
        result = result.replace('<th>', '<th style=" border: 1px solid silver; text-align: center;">')
        self.ketqua.setHtml(result)
        self.list = []
        self.string = []
        self.index = 0


    def getdf(self):
        return self.df

    def showMessageBox(self, text):
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Warning)
        msgBox.setWindowTitle("Cảnh báo!")
        msgBox.setText(text)
        msgBox.setStandardButtons(QtWidgets.QMessageBox.Ok)
        x = msgBox.exec_()

    def check_flow(self) -> object:
        print(self.getdf())
        if self.getdf() is None:
            return False
        else:
            return True

    def roirac(self):
        str = self.input_roirac.text()
        # print(str)
        # print(self.check_flow())

        check_flow = self.check_flow()
        print(check_flow)
        if check_flow == False:
            self.showMessageBox("Vui lòng chọn dữ liệu")
        else:
            df = self.getdf()
            if (str == 'pdays'):
                maxpdays = df[str].max()
                interval = (maxpdays) / 3
                bin1 = interval
                bin2 = bin1 + interval
                print(bin1, bin2)

                for dataset in [df]:
                    dataset[str] = dataset[str].astype(int)
                    dataset.loc[dataset[str] == -1, str] = 0
                    dataset.loc[(dataset[str] > 0) & (dataset[str] <= bin1), str] = 1
                    dataset.loc[(dataset[str] > bin1) & (dataset[str] <= bin2), str] = 2
                    dataset.loc[(dataset[str] > bin2), str] = 3
            if (str == 'duration'):
                for dataset in [df]:
                    dataset[str] = dataset[str].astype(int)
                    dataset.loc[dataset[str] <= 104.0, str] = 0
                    dataset.loc[(dataset[str] > 104.0) & (dataset[str] <= 185.0), str] = 1
                    dataset.loc[(dataset[str] > 185.0) & (dataset[str] <= 329.0), str] = 2
                    dataset.loc[(dataset[str] > 329.0), str] = 3
            if (str == 'balance'):
                for dataset in [df]:
                    dataset[str] = dataset[str].astype(int)
                    dataset.loc[dataset[str] <= 69.0, str] = 0
                    dataset.loc[(dataset[str] > 69.0) & (dataset[str] <= 444.0), str] = 1
                    dataset.loc[(dataset[str] > 444.0) & (dataset[str] <= 1480.0), str] = 2
                    dataset.loc[(dataset[str] > 1480.0), str] = 3
            else:
                min = df[str].min()
                max = df[str].max()
                interval = (max - min) / 4
                bin1 = min + interval
                bin2 = bin1 + interval
                bin3 = bin2 + interval
                # print(bin1, bin2, minage, interval)
                for dataset in [df]:
                    dataset[str] = dataset[str].astype(int)
                    dataset.loc[dataset[str] <= bin1, str] = 0
                    dataset.loc[(dataset[str] > bin1) & (dataset[str] <= bin2), str] = 1
                    dataset.loc[(dataset[str] > bin2) & (dataset[str] <= bin3), str] = 2
                    dataset.loc[(dataset[str] > bin3), str] = 3
            result = df[str].value_counts()
            s = result.rename_axis('nhãn giá trị').reset_index(name='số dòng')
            show = s.to_html(justify='center', border=0)
            show = show.replace('<table border="0" class="dataframe">',
                                '<table style="text-align: center; font-size: 11pt; font-family: Arial; ">')
            show = show.replace('<td>', '<td style=" border: 1px solid silver;text-align: center;">')
            show = show.replace('<th>',
                                '<th style=" border: 1px solid silver; text-align: center;background: #E0E0E0;">')
            show = show.replace('<thead>', '<thead style="background: #E0E0E0; padding: 5px;">')
            show = show.replace('<tr>', '<tr style=" :hover {background: silver; cursor: pointer;}">')

            print(show)

            self.ketqua.clear()
            self.ketqua.setHtml(show)
            self.input_roirac.clear()

    def xoa(self):
        str = self.input_xoa.text()
        check_flow = self.check_flow()
        print(check_flow)
        if check_flow == False:
            self.showMessageBox("Vui lòng chọn dữ liệu")
        else:

            self.df = self.getdf()
            self.df = self.df.drop([str], axis=1)
            result = self.df.head(30).to_html(border=0)
            show = result.replace('<table border="0" class="dataframe">',
                                  '<table style="text-align: center; font-size: 11pt; font-family: Arial; ">')
            show = show.replace('<td>', '<td style=" border: 1px solid silver;text-align: center;">')
            show = show.replace('<th>','<th style=" border: 1px solid silver; text-align: center;background: #E0E0E0;">')
            show = show.replace('<thead>', '<thead style="background: #E0E0E0; padding: 5px;">')

            # self.ketqua.clear()
            self.ketqua.setHtml(show)
            self.input_xoa.clear()

    def thong_ke(self):
        check_flow = self.check_flow()
        print(check_flow)
        if check_flow == False:
            self.showMessageBox("Vui lòng chọn dữ liệu")
        else:
            input = self.input_thongke.text()
            df = self.getdf()
            sns.barplot(x="y", y=input, data=df)
            plt.show()


    def getstring(self):
        return self.string

    def getlist(self):
        return self.list

    def uncheck(self, state):
        if state == QtCore.Qt.CheckState:
            sender = self.sender()
            print(sender)
            if sender ==self.ID3_check:
                self.C45_check.setChecked(False)
            elif sender ==self.C45_check:
                self.ID3_check.setChecked(False)

    def cal_matrix(self):
        check_flow = self.check_flow()
        print(check_flow)
        if check_flow == False:
            self.showMessageBox("Vui lòng chọn dữ liệu")
        else:
            train_size =  int(self.input_training.text())
            print(train_size)
            df = self.getdf()
            test_size = 1 - (train_size / 100)
            # tach cot du lieuj
            features = df.drop('y', axis=1)
            labels = df['y']

            # chuyen ve dang one-hot
            features.select_dtypes(exclude=['int64']).columns
            features_onehot = pd.get_dummies(features, columns=features.select_dtypes(exclude=['int64']).columns)

            # tach thanhf test va train
            x_train, x_test, y_train, y_test = train_test_split(features_onehot, labels, test_size=test_size,
                                                                random_state=42)

            if (self.C45_check.isChecked()):
                self.clf = tree.DecisionTreeClassifier(criterion="gini", random_state=0)
            if (self.ID3_check.isChecked()):
                self.clf = tree.DecisionTreeClassifier(criterion="entropy", random_state=0)

            # train decision tree
            self.clf.fit(x_train, y_train)
            # duự đoán test
            self.tree_pred = self.clf.predict(x_test)

            # độ chính xác
            tree_score = metrics.accuracy_score(y_test, self.tree_pred)

            report = metrics.classification_report(y_test, self.tree_pred)
            # ma tran nham lan
            tree_cm = metrics.confusion_matrix(y_test, self.tree_pred)

            result = "Test size: " + str(test_size) + '\n' + "Accuracy: " + str(tree_score)  + '\n' +"Report: " + str(
                report) + '\n' + "Confusion matrix: " + '\n' + str(tree_cm)
            self.ketqua.setText(result)

            # luu ket qua vao combobox
            self.string = self.getstring()
            self.string.append(result)

            self.list = self.getlist()
            self.list.append(self.index)

            self.cbb_save.addItem(f"Lần chạy thứ {str(self.index)}")

            self.index = self.index + 1

            # bieu dien ma tran nham lan
            plt.figure(figsize=(7, 7))
            sns.heatmap(tree_cm, annot=True, fmt=".3f", linewidths=.4, square=True, cmap='Blues_r')
            plt.xlabel('Lớp dự đoán từ mô hình')
            plt.ylabel('Lớp trên thực tế')
            if (self.C45_check.isChecked()):
                title = "Độ chính xác của cây quyết định CART: {0}".format(tree_score)
            if (self.ID3_check.isChecked()):
                title = "Độ chính xác của cây quyết định ID3: {0}".format(tree_score)
            plt.title(title, size=10)
            plt.show()

    def index_changed_callback(self, state):
        print(state)
        self.ketqua.setText(self.string[state])


    def draw(self):
        check_flow = self.check_flow()
        print(check_flow)
        if check_flow == False:
            self.showMessageBox("Vui lòng chọn dữ liệu")
        else:
            text_representation = tree.export_text(self.clf)
            print(text_representation)
            self.ketqua.setText(text_representation)
            fig, ax = plt.subplots(figsize=(50, 24))
            tree.plot_tree(self.clf, filled=True, fontsize=10)
            plt.savefig('decision_tree', dpi=100)
            plt.show()

    def clear(self):
        self.setupUi(MainWindow)


if __name__ == '__main__':
    import sys

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
