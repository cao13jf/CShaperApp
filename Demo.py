
from CShaper import Ui_MainWindow
from PyQt5.QtWidgets import (QApplication, QGridLayout, QGroupBox, QDialog, QTableView,
                             QLabel, QSlider, QVBoxLayout, QMainWindow, QLineEdit,
                             QMessageBox, QComboBox, QTableWidgetItem, QAbstractItemView)
from PyQt5 import QtWidgets
from PyQt5.QtGui import QStandardItemModel,QStandardItem
from FuncThread import PreprocessThread, SegmentationThread, AnalysisThread
import warnings
from multiprocessing import freeze_support
import re
import sys
import time
from ShapeUtil.data_structure import *
import subprocess
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")


class MainForm(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainForm, self).__init__()
        self.setupUi(self)
        self.dirNameView = ''
        self.Function.currentChanged.connect(self.updateBlankInfo)
        self.tabWidget.currentChanged.connect(self.updateDataTable)
        self.t3 = self.tableView_3.frameGeometry()
        self.t3.setY(self.t3.y() + 30)

        # self.tableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)

        # combine_slice.py
        self.Btn_rawFolder.clicked.connect(self.chooseRawFolder_Pre)
        self.Btn_projectFolder.clicked.connect(self.chooseProjectFolder_Pre)
        self.Btn_lineageFile.clicked.connect(self.chooseLineageFile_Pre)
        self.Btn_runPreprocess.clicked.connect(self.runPreprocess)
        self.actionRun_Preprocess.triggered.connect(self.runPreprocess)
        self.Btn_numberDict.clicked.connect(self.chooseNumberDict)
        self.Btn_stopPreprocess.clicked.connect(self.stopPreprocess)

        # self.LE_rawFolder.setText('/Users/admin/cuhk/CShaperAPP/Data/MembRaw')
        # self.CB_embryoNames.setCurrentText('181210plc1p1')
        # self.LE_xyResolution.setText('0.09')
        # self.LE_zResolution.setText('0.42')
        # self.LE_reduceRatio.setText('0.3')
        # self.LE_sliceNum.setText('68')
        # self.LE_maxTime.setText('5')
        # self.LE_projectFolder.setText('/Users/admin/cuhk/CShaperAPP/TestProject')
        # self.LE_lineage.setText('/Users/admin/cuhk/CShaperAPP/Data/MembRaw/181210plc1p1/aceNuc/CD181210plc1p1.csv')
        # self.LE_numberDict.setText('/Users/admin/cuhk/CShaperAPP/Resource/number_dictionary.csv')

        # test_edt.py
        self.Btn_projectFolder_Seg.clicked.connect(self.chooseProjectFolder_Seg)
        self.Btn_modelFile_Seg.clicked.connect(self.chooseModelFile_Seg)
        self.Btn_runSegmentation.clicked.connect(self.runSegmentation)
        self.actionRun_Segmentation.triggered.connect(self.runSegmentation)
        self.Btn_stopSegmentation.clicked.connect(self.stopSegmentation)

        # shape_analysis.py
        self.Btn_runAnalysis.clicked.connect(self.runAnalysis)
        self.actionRun_Analysis.triggered.connect(self.runAnalysis)
        self.Btn_numberDict_Ana.clicked.connect(self.chooseNumberDict_Ana)
        self.Btn_rawFolder_Ana.clicked.connect(self.chooseRawFolder_Ana)
        self.Btn_projectFolder_Ana.clicked.connect(self.chooseProjectFolder_Ana)
        self.Btn_lineageFile_Ana.clicked.connect(self.chooseLineageFile_Ana)
        self.Btn_stopAnalysis.clicked.connect(self.stopAnalysis)

        # run all
        self.Btn_runAll.clicked.connect(self.runAll)
        self.actionRun_ALL.triggered.connect(self.runAll)

        #action File
        self.actionNew_Project.triggered.connect(self.newProjoect)
        self.actionSave_Project.triggered.connect(self.saveProject)
        self.actionLoad_Project.triggered.connect(self.loadProject)

        self.actionOpen_Result_Folder.triggered.connect(self.openResultFolder)
        # PyQt5.QtWidgets.QUndoCommand
        #action Edit
        self.actionUndo.triggered.connect(self.undoEdit)
        self.actionRedo.triggered.connect(self.redoEdit)
        self.actionCopy.triggered.connect(self.copyEdit)
        self.actionPaste.triggered.connect(self.pasteEdit)

        #action About
        self.actionCopy_Right.triggered.connect(self.copyRight)
        self.actionHelp.triggered.connect(self.helpAbout)
        self.actionVersion.triggered.connect(self.versionAbout)

    def updateBlankInfo(self):
        if self.Function.currentIndex() == 0:
            if self.LE_rawFolder.text() != '':
                self.CB_embryoNames.clear()
                listdir = os.listdir(self.LE_rawFolder.text())
                listdir.sort()
                self.CB_embryoNames.addItems(listdir)
        if self.Function.currentIndex() == 1:
            if self.LE_projectFolder.text() != '':
                self.LE_projectFolder_Seg.setText(self.LE_projectFolder.text())
                try:
                    self.CB_embryoNames_Seg.clear()
                    if os.path.isdir(os.path.join(self.LE_projectFolder.text(), "RawStack")):
                        self.CB_embryoNames_Seg.clear()
                        listdir = os.listdir(os.path.join(self.LE_projectFolder.text(), "RawStack"))
                        listdir.sort()
                        self.CB_embryoNames_Seg.addItems(listdir)
                    else:
                        os.makedirs(os.path.join(self.LE_projectFolder.text(), "RawStack"))
                except Exception:
                    QMessageBox.warning(self, 'Warning!', 'Folder Error, Please check it!')
            if self.LE_maxTime.text() != '':
                self.LE_maxTime_Seg.setText(self.LE_maxTime.text())
        if self.Function.currentIndex() == 2:
            if self.LE_rawFolder.text() != '':
                self.LE_rawFolder_Ana.setText(self.LE_rawFolder.text())
                try:
                    self.CB_embryoNames_Ana.clear()
                    listdir = os.listdir(self.LE_rawFolder.text())
                    listdir.sort()
                    self.CB_embryoNames_Ana.addItems(listdir)
                except Exception:
                    QMessageBox.warning(self, 'Warning!', 'Folder Error, Please check it!')
            if self.LE_xyResolution.text() != '':
                self.LE_xyResolution_Ana.setText(self.LE_xyResolution.text())
            if self.LE_sliceNum.text() != '':
                self.LE_sliceNum_Ana.setText(self.LE_sliceNum.text())
            if self.LE_projectFolder_Seg.text() != '':
                self.LE_projectFolder_Ana.setText(self.LE_projectFolder_Seg.text())
            elif self.LE_projectFolder.text() != '':
                self.LE_projectFolder_Ana.setText(self.LE_projectFolder.text())
            if self.LE_lineage.text() != '':
                self.LE_lineage_Ana.setText(self.LE_lineage.text())
        # print( self.Function.currentIndex())
        if self.Function.currentIndex() == 3:
            if self.dirNameView != '':
                try:
                    r = os.listdir(self.dirNameView)
                    for i in r:
                        if i.endswith('surface.csv'):
                            file = self.dirNameView + '/' + i
                    # file = '/Users/admin/cuhk/CShaperAPP/TestProject/StatShape/181210plc1p1/181210plc1p1_surface.csv'
                    self.showDataTable(file)
                except Exception:
                    QMessageBox.warning(self, 'Error!', 'Folder Error!')
            else:
                try:
                    self.dirNameView = QtWidgets.QFileDialog.getExistingDirectory(self, 'Choose Project Folder', './')
                    r = os.listdir(self.dirNameView)
                    for i in r:
                        if i.endswith('surface.csv'):
                            folder = self.dirNameView + '/' + i
                    self.showDataTable(folder)
                except Exception:
                    QMessageBox.warning(self, 'Error!', 'Folder Error!')

    def showDataTable(self, filename):
        input_table = pd.read_csv(filename)
        input_table_rows = input_table.shape[0]
        input_table_colunms = input_table.shape[1]

        data = input_table.values.tolist()

        self.tableView_3.close()
        self.tableView_3 = QTableView(self.tabWidget)
        self.tableView_3.setGeometry(self.t3)
        self.Model = QStandardItemModel()
        self.Model.setHorizontalHeaderLabels(input_table)
        for i in range(input_table_rows):
            for j in range(input_table_colunms):
                self.Model.setItem(i, j, QStandardItem(str(data[i][j])))

        self.tableView_3.setModel(self.Model)
        self.tableView_3.updateEditorData()
        self.tableView_3.show()


    def updateDataTable(self):
        filename = ''
        try:
            r = os.listdir(self.dirNameView)
            if self.tabWidget.currentIndex() == 0:
                for i in r:
                    if i.endswith('surface.csv'):
                        filename = self.dirNameView + '/' + i
                self.showDataTable(filename)
            elif self.tabWidget.currentIndex() == 1:
                for i in r:
                    if i.endswith('volume.csv'):
                        filename = self.dirNameView + '/' + i
                self.showDataTable(filename)
            elif self.tabWidget.currentIndex() == 2:
                for i in r:
                    if i.endswith('contact.csv'):
                        filename = self.dirNameView + '/' + i
                self.showDataTable(filename)
        except Exception:
            pass

    def runAll(self):
        self.runPreprocess()
        self.runSegmentation()
        self.runAnalysis()

    def chooseRawFolder_Pre(self):
        dirName = QtWidgets.QFileDialog.getExistingDirectory(self, 'Choose Raw Folder', './')
        try:
            self.CB_embryoNames.clear()
            self.LE_rawFolder.setText(dirName)
            listdir = os.listdir(dirName)
            listdir.sort()
            self.CB_embryoNames.addItems(listdir)

        except Exception as e:
            QMessageBox.warning(self, 'Warning!', 'Please Choose Right Folder!')

    def chooseProjectFolder_Pre(self):
        dirName = QtWidgets.QFileDialog.getExistingDirectory(self, 'Choose Stack Folder', './')
        try:
            self.LE_projectFolder.setText(dirName)
        except Exception as e:
            QMessageBox.warning(self, 'Warning!', 'Please Choose Right Folder!')

    def chooseLineageFile_Pre(self):
        fileName, fileType = QtWidgets.QFileDialog.getOpenFileName(self, 'Choose Lineage File', self.LE_rawFolder.text(), "CSV Files(*.csv)")
        try:
            self.LE_lineage.setText(fileName)
        except Exception as e:
            QMessageBox.warning(self, 'Warning!', 'Please Choose Right Folder!')
    def chooseNumberDict(self):
        fileName, fileType = QtWidgets.QFileDialog.getOpenFileName(self, 'Choose Lineage File',
                                                                   './', "CSV Files(*.csv)")
        try:
            self.LE_numberDict.setText(fileName)
        except Exception as e:
            QMessageBox.warning(self, 'Warning!', 'Please Choose Right Folder!')

    def runPreprocess(self):
        config = {}
        try:
            config['num_slice'] = int(self.LE_sliceNum.text())
            en = []
            en.append(self.CB_embryoNames.currentText())
            config["embryo_names"] = en
            config["max_time"] = int(self.LE_maxTime.text())
            config["xy_resolution"] = float(self.LE_xyResolution.text())
            config["z_resolution"] = float(self.LE_zResolution.text())
            config["reduce_ratio"] = float(self.LE_reduceRatio.text())
            config["raw_folder"] = self.LE_rawFolder.text()
            config["project_folder"] = self.LE_projectFolder.text()
            config["lineage_file"] = self.LE_lineage.text()
            config["number_dictionary"] = self.LE_numberDict.text()
        except Exception:
            QMessageBox.warning(self, 'Error!', 'Please check your paras!')
        self.LE_maxTime_Seg.setText(self.LE_maxTime.text())
        self.LE_sliceNum_Ana.setText(self.LE_sliceNum.text())
        self.call = False
        self.pthread = PreprocessThread(config)
        self.pthread.signal.connect(self.ThreadCallback)
        self.pthread.process.connect(self.ProcessCallback)
        self.pthread.start()

    def stopPreprocess(self):
        try:
            self.pthread.terminate()
            QMessageBox.information(self, 'Tips', 'Preprocess has been terminated.')
        except Exception:
            QMessageBox.warning(self, 'Warning!', 'Preprocess has not been started.')

    def ThreadCallback(self, call, func):

        if call == True:
            if func == 'Preprocess':
                self.PreprogressBar.setValue(100)
            elif func == 'Segmentation':
                self.SegmentationBar.setValue(100)
            elif func == 'Analysis':
                self.AnalysisBar.setValue(100)
            QMessageBox.information(self, func, func+' success!')
        elif call == False:
            QMessageBox.warning(self, 'Error!', func+' failed!')
        else:
            pass

    def ProcessCallback(self, func, current, max_time):
        self.label_Preprocess.setText(func+':')
        self.PreprogressBar.setValue((current) * 100 / (max_time * 3))

    def chooseProjectFolder_Seg(self):
        dirName = QtWidgets.QFileDialog.getExistingDirectory(self, 'Choose Stack Folder', './')
        try:
            self.LE_projectFolder_Seg.setText(dirName)
            # max_time = len(os.listdir(os.path.join(dirName, "RawStack")))
            if os.path.isdir(os.path.join(dirName, "RawStack")):
                self.CB_embryoNames_Seg.clear()
                listdir = os.listdir(os.path.join(dirName, "RawStack"))
                listdir.sort()
                self.CB_embryoNames_Seg.addItems(listdir)
            else:
                os.makedirs(os.path.join(dirName, "RawStack"))
        except Exception as e:
            QMessageBox.warning(self, 'Warning!', 'Please Choose Right Folder!')

    def chooseModelFile_Seg(self):
        fileName, fileType = QtWidgets.QFileDialog.getOpenFileName(self, 'Choose Model File',
                                                                   './', "Model Files(*.ckpt.*)")
        try:
            model_name = re.findall(r'^.*.ckpt',fileName)
            self.LE_modelFile_Seg.setText(model_name[0])
        except Exception as e:
            QMessageBox.warning(self, 'Warning!', 'Please Choose Right Model!')

    def runSegmentation(self):
        config = {}
        try:
            config['para']={}
            config["para"]["project_folder"] = self.LE_projectFolder_Seg.text()
            en = []
            en.append(self.CB_embryoNames_Seg.currentText())
            config["para"]["embryo_names"] = en
            config["para"]["max_time"] = int(self.LE_maxTime_Seg.text())
            # config["para"]["save_folder"] = self.LE_saveFolder_Seg.text()
            config["para"]["batch_size"] = int(self.LE_batchSize_Seg.text())
            lineage = self.CB_lineage_Seg.currentText()
            if lineage == 'No lineage':
                config["para"]["nucleus_as_seed"] = False
                config["para"]["nucleus_filter"] = False
            elif lineage == 'Before segmentation':
                config["para"]["nucleus_as_seed"] = True
                config["para"]["nucleus_filter"] = False
            elif lineage == 'After segmentation':
                config["para"]["nucleus_as_seed"] = False
                config["para"]["nucleus_filter"] = True

            config["data"] = {}
            config["data"]["data_root"] = os.path.join(config["para"]["project_folder"], "RawStack")
            config["data"]["data_names"] = config["para"]["embryo_names"]
            config["data"]["max_time"] = config["para"]["max_time"]
            config["data"]["save_folder"] = os.path.join(config["para"]["project_folder"], "CellMembrane")

            config["data"]["with_ground_truth"] = False
            config["data"]["label_edt_transform"] = True
            config["data"]["valid_edt_width"] = 30
            config["data"]["label_edt_discrete"] = True
            config["data"]["edt_discrete_num"] = 16

            config["network"] = {}
            config["network"]["net_type"] = "CShaper"
            config["network"]["net_name"] = "DMapNet_PUB"
            config["network"]["data_shape"] = [24, 128, 96, 1]
            config["network"]["label_shape"] = [16, 128, 96, 1]
            config["network"]["model_file"] = self.LE_modelFile_Seg.text()

            config["testing"] = {}
            config["testing"]["batch_size"] = config["para"]["batch_size"]
            config["testing"]["nucleus_as_seed"] = config["para"]["nucleus_as_seed"]
            config["testing"]["nucleus_filter"] = config["para"]["nucleus_filter"]
            config["testing"]["save_binary_seg"] = True
            config["testing"]["save_predicted_map"] = False
            config["testing"]["slice_direction"] = "sagittal"
            config["testing"]["direction_fusion"] = True
            config["testing"]["only_post_process"] = False
            config["testing"]["post_process"] = True

            config["segdata"] = {}
            config["segdata"]["membseg_path"] = config["data"]["save_folder"]
            config["segdata"]["nucleus_data_root"] = config["data"]["data_root"]

            config["debug"] = {}
            config["debug"]["debug_mode"] = False
            config["debug"]["save_anisotropic"] = False
            config["debug"]["save_graph_model"] = False
            config["debug"]["save_init_watershed"] = False
            config["debug"]["save_merged_seg"] = False
            config["debug"]["save_cell_nomemb"] = False
        except Exception:
            QMessageBox.warning(self, 'Error!', 'Please check your paras!')
        self.call = False
        self.sthread = SegmentationThread(config)
        self.sthread.signal.connect(self.ThreadCallback)
        self.sthread.process.connect(self.SegmentationCallback)
        self.sthread.start()

    def SegmentationCallback(self, func, current, max_time):
        self.label_Segmentation.setText(func+':')
        self.SegmentationBar.setValue((current) * 100 / (max_time * 2))

    def stopSegmentation(self):
        try:
            self.sthread.terminate()
            QMessageBox.information(self, 'Tips', 'Segmentation has been terminated.')
        except Exception:
            QMessageBox.warning(self, 'Warning!', 'Segmentation has not been started.')

    def chooseRawFolder_Ana(self):
        dirName = QtWidgets.QFileDialog.getExistingDirectory(self, 'Choose Raw Folder', './')
        try:
            self.CB_embryoNames_Ana.clear()
            self.LE_rawFolder_Ana.setText(dirName)
            listdir = os.listdir(dirName)
            listdir.sort()
            self.CB_embryoNames_Ana.addItems(listdir)
        except Exception as e:
            QMessageBox.warning(self, 'Warning!', 'Please Choose Right Folder!')

    def chooseProjectFolder_Ana(self):
        dirName = QtWidgets.QFileDialog.getExistingDirectory(self, 'Choose Raw Folder', './')
        try:
            self.LE_projectFolder_Ana.setText(dirName)
        except Exception as e:
            QMessageBox.warning(self, 'Warning!', 'Please Choose Right Folder!')

    def chooseNumberDict_Ana(self):
        fileName, fileType = QtWidgets.QFileDialog.getOpenFileName(self, 'Choose Lineage File',
                                                                   './', "CSV Files(*.csv)")
        try:
            self.LE_numberDict_Ana.setText(fileName)
        except Exception as e:
            QMessageBox.warning(self, 'Warning!', 'Please Choose Right Folder!')

    def chooseLineageFile_Ana(self):
        fileName, fileType = QtWidgets.QFileDialog.getOpenFileName(self, 'Choose Lineage File', './', "CSV Files(*.csv)")
        try:
            self.LE_lineage_Ana.setText(fileName)
        except Exception as e:
            QMessageBox.warning(self, 'Warning!', 'Please Choose Right Folder!')

    def runAnalysis(self):

        config = {}
        try:
            config['para'] = {}
            config['para']['num_slice'] = int(self.LE_sliceNum_Ana.text())
            config['para']['xy_resolution'] = float(self.LE_xyResolution_Ana.text())
            config['para']['raw_folder'] = self.LE_rawFolder_Ana.text()
            en = []
            en.append(self.CB_embryoNames_Ana.currentText())
            config["para"]["embryo_names"] = en
            config['para']['project_folder'] = self.LE_projectFolder_Ana.text()
            config['para']['first_run'] = False
            config['para']["number_dictionary"] = self.LE_numberDict_Ana.text()
            config['para']["lineage_file"] = self.LE_lineage_Ana.text()
            self.dirNameView = self.LE_projectFolder_Ana.text() + '/StateShape/' + self.CB_embryoNames_Ana.currentText()
        except Exception:
            QMessageBox.warning(self, 'Error!', 'Please check your paras!')
        self.call = False
        self.athread = AnalysisThread(config)
        self.athread.signal.connect(self.ThreadCallback)
        self.athread.process.connect(self.AnalysisCallback)
        self.athread.start()

    def AnalysisCallback(self, func, current, max_time):
        self.label_Analysis.setText(func+':')
        self.AnalysisBar.setValue((current) * 100 / (max_time))

    def stopAnalysis(self):
        try:
            self.athread.terminate()
            QMessageBox.information(self, 'Tips', 'Analysis has been terminated.')
        except Exception:
            QMessageBox.warning(self, 'Warning!', 'Analysis has not been started.')

    def newProjoect(self):
        for i in self.findChildren(QLineEdit):
            i.setText('')

        for i in self.findChildren(QComboBox):
            i.clear()

    def saveProject(self):
        dirName = QtWidgets.QFileDialog.getExistingDirectory(self, 'Choose Save Folder', './')
        try:
            # save all the paras
            with open(dirName+'/test.project', 'w', encoding='utf8', newline='') as fout:

                for i in self.findChildren(QLineEdit):
                    fout.write(i.objectName() + ':' + i.text() + '\n')
                for i in self.findChildren(QComboBox):
                    fout.write(i.objectName() + ':' + i.currentText() + '\n')
        except Exception as e:
            QMessageBox.warning(self, 'Warning!', 'Project Save Failed!')

    def loadProject(self):
        fileName, fileType = QtWidgets.QFileDialog.getOpenFileName(self, 'Choose Project File', './', "Project File(*.project)")

        try:
            with open(fileName, 'r', encoding='utf8') as fr:
                r = fr.readlines()
                for i in r:
                    temp = i.split(':')
                    temp[1] = temp[1].strip('\n')
                    if self.findChild(QLineEdit,temp[0]) is not None:
                        self.findChild(QLineEdit, temp[0]).setText(temp[1])
                    elif self.findChild(QComboBox, temp[0]) is not None:
                        self.findChild(QComboBox, temp[0]).setCurrentText(temp[1])
        except Exception:
            QMessageBox.warning(self, 'Warning!', 'Project Load Failed!')

    def undoEdit(self):
        self.focusWidget().undo()

    def redoEdit(self):
        self.focusWidget().redo()

    def copyEdit(self):
        self.focusWidget().copy()

    def pasteEdit(self):
        self.focusWidget().paste()

    def openResultFolder(self):
        try:
            if self.LE_projectFolder.text() != '':
                subprocess.call(['open',self.LE_projectFolder.text()])
        except Exception:
            QMessageBox.warning(self, 'Warning!', 'Open Result Folder Failed!')

    def versionAbout(self):
        QMessageBox.information(self, 'Version', 'The version of software is 1.0.0 alpha.')

    def helpAbout(self):
        QMessageBox.information(self, 'Help', 'This software is about ... our github is https://.....')

    def copyRight(self):
        QMessageBox.information(self, 'Copy Right', 'Here is the copy right information of our software.')


if __name__ == '__main__':
    freeze_support()
    app = QtWidgets.QApplication(sys.argv)
    win = MainForm()
    win.show()
    sys.exit(app.exec())