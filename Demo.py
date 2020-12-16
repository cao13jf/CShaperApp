
from CShaper import Ui_MainWindow
from about import Ui_Dialog
from PyQt5.QtWidgets import (QApplication, QGridLayout, QGroupBox, QDialog, QTableView,
                             QLabel, QSlider, QVBoxLayout, QMainWindow, QLineEdit,
                             QMessageBox, QComboBox, QTableWidgetItem, QAbstractItemView)
from PyQt5 import QtWidgets
from PyQt5.QtGui import QStandardItemModel, QStandardItem
import PyQt5.QtCore
from FuncThread import PreprocessThread, SegmentationThread, AnalysisThread, RunAllThread
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import vtkmodules.all as vtk

from reconstruction import *
import SimpleITK as sitk
from random import random

import warnings
from multiprocessing import freeze_support
import re
import sys
import time
from ShapeUtil.data_structure import *
import subprocess
import pandas as pd
import numpy as np
import platform
import webbrowser

warnings.filterwarnings("ignore")
MASK_OPACITY = 1
MASK_SMOOTHNESS = 500


class ChildDialog(QDialog, Ui_Dialog):
    def __init__(self):
        QDialog.__init__(self)
        self.child = Ui_Dialog()  # 子窗口的实例化

        self.setWindowIcon(PyQt5.QtGui.QIcon('CShaperLogo.png'))
        self.child.setupUi(self)


class MainForm(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainForm, self).__init__()
        self.setWindowTitle('CShaper')
        self.setWindowIcon(PyQt5.QtGui.QIcon('CShaperLogo.png'))
        self.setupUi(self)
        self.dirNameView = ''
        self.Function.currentChanged.connect(self.updateBlankInfo)
        self.tabWidget.currentChanged.connect(self.updateDataTable)
        self.t3 = self.tableView_3.frameGeometry()
        self.t3.setY(self.t3.y() + 30)

        # 3d reconstruction
        self.reconstructView = ''
        self.embryo = ''
        self.main_widget = QtWidgets.QWidget(self)
        self.vtkWidget = QVTKRenderWindowInteractor(self.main_widget)
        self.gridLayout.addWidget(self.vtkWidget)

        renderer = vtk.vtkRenderer()
        renderer.SetBackground(1, 1, 1)
        self.renderlist = []
        self.MASK_COLORS = []
        self.render_window = self.vtkWidget.GetRenderWindow()
        self.render_window.AddRenderer(renderer)
        self.render_window.Render()
        self.iren = self.render_window.GetInteractor()

        self.x = 1
        self.maxNum = 1
        self.Label_idx.setWordWrap(True)
        self.Label_idx.setAlignment(PyQt5.QtCore.Qt.AlignTop)
        self.Slider_3d.setMinimum(1)
        # self.Slider_3d.sliderReleased.connect(self.set3DImage)
        self.Slider_3d.valueChanged.connect(self.set3DImage)

        # combine_slice.py
        self.Btn_rawFolder.clicked.connect(self.chooseRawFolder_Pre)
        self.Btn_projectFolder.clicked.connect(self.chooseProjectFolder_Pre)
        self.Btn_lineageFile.clicked.connect(self.chooseLineageFile_Pre)
        self.Btn_runPreprocess.clicked.connect(self.runPreprocess)
        self.actionRun_Preprocess.triggered.connect(self.runPreprocess)
        self.Btn_numberDict.clicked.connect(self.chooseNumberDict)
        self.Btn_stopPreprocess.clicked.connect(self.stopPreprocess)

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

        # action File
        self.actionNew_Project.triggered.connect(self.newProjoect)
        self.actionSave_Project.triggered.connect(self.saveProject)
        self.actionLoad_Project.triggered.connect(self.loadProject)

        self.actionOpen_Result_Folder.triggered.connect(self.openResultFolder)
        # PyQt5.QtWidgets.QUndoCommand
        # action Edit
        self.actionUndo.triggered.connect(self.undoEdit)
        self.actionRedo.triggered.connect(self.redoEdit)
        self.actionCopy.triggered.connect(self.copyEdit)
        self.actionPaste.triggered.connect(self.pasteEdit)

        # action About
        # self.actionCopy_Right.triggered.connect(self.copyRight)
        # child = ChildDialog()
        # self.actionHelp.triggered.connect(child.show)
        # self.actionHelp.triggered.connect(self.helpAbout)
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
            if self.LE_numberDict.text() != '':
                self.LE_numberDict_Ana.setText(self.LE_numberDict.text())
        if self.Function.currentIndex() == 3:
            if self.dirNameView != '':
                try:
                    r = os.listdir(self.dirNameView)
                    for i in r:
                        if i.endswith('surface.csv'):
                            file = self.dirNameView + '/' + i
                            break
                    self.showDataTable(file, self.tableView_3)
                except Exception:
                    QMessageBox.warning(self, 'Error!', 'Folder Error!')

    def showDataTable(self, filename, tableView):
        input_table = pd.read_csv(filename,header=None)
        input_table_rows = input_table.shape[0]
        input_table_colunms = input_table.shape[1]

        data = input_table.values.tolist()
        self.Model = QStandardItemModel()
        for i in range(input_table_rows):
            for j in range(input_table_colunms):
                if str(data[i][j]) == 'nan':
                    self.Model.setItem(i, j, QStandardItem(''))
                else:
                    try:
                        self.Model.setItem(i, j, QStandardItem(str(round(data[i][j], 2))))
                    except Exception:
                        self.Model.setItem(i, j, QStandardItem(str(data[i][j])))
        tableView.verticalHeader().hide()
        tableView.horizontalHeader().hide()
        tableView.setModel(self.Model)
        tableView.updateEditorData()
        tableView.show()

    def construct3D(self, file):
        renderer = vtk.vtkRenderer()
        renderer.SetBackground(1, 1, 1)
        mask = NiiObject()
        mask.reader = vtk.vtkNIFTIImageReader()
        mask.reader.SetFileName(file)
        mask.reader.Update()
        mask.extent = mask.reader.GetDataExtent()
        n_labels = int(mask.reader.GetOutput().GetScalarRange()[1])
        image = sitk.ReadImage(file)
        image = sitk.GetArrayFromImage(image)
        ccc = {}
        for i in image:
            for j in i:
                for k in j:
                    if k in ccc.keys():
                        ccc[k] += 1
                    else:
                        ccc[k] = 1

        count = 0
        for label_idx in range(n_labels + 1):
            if count>len(self.MASK_COLORS)-1:
                for i in range(count-len(self.MASK_COLORS)+1):
                    temp = (random(), random(), random())
                    self.MASK_COLORS.append(temp)

            if label_idx in ccc.keys():
                if label_idx > 0:
                    mask.labels.append(NiiLabel(self.MASK_COLORS[count], MASK_OPACITY, MASK_SMOOTHNESS))
                    mask.labels[count].extractor = create_mask_extractor(mask)
                    add_surface_rendering(mask, count, label_idx)
                    renderer.AddActor(mask.labels[count].actor)
                    count += 1
                mask.labels.append(NiiLabel(self.MASK_COLORS[count], MASK_OPACITY, MASK_SMOOTHNESS))
                mask.labels[count].extractor = create_mask_extractor(mask)
                add_surface_rendering(mask, count, label_idx + 1)
                renderer.AddActor(mask.labels[count].actor)
                count += 1
                if label_idx < n_labels:
                    mask.labels.append(NiiLabel(self.MASK_COLORS[count], MASK_OPACITY, MASK_SMOOTHNESS))
                    mask.labels[count].extractor = create_mask_extractor(mask)
                    add_surface_rendering(mask, count, label_idx + 2)
                    renderer.AddActor(mask.labels[count].actor)
                    count += 1
        renderer.ResetCamera()
        return renderer


    def create3DImage(self):
        self.renderlist = []
        self.MASK_COLORS = []
        for i in range(1000):
            temp = (random(), random(), random())
            self.MASK_COLORS.append(temp)
        for i in range(self.maxNum):
            file = self.reconstructView + self.embryo + '_' + str('%03d' % (i+1)) + '_segCell.nii.gz'
            self.renderlist.append(self.construct3D(file))

    def set3DImage(self, x):
        self.Label_idx.setText('{}/{}'.format('%03d' % x, '%03d' % self.maxNum))
        image = self.renderlist[x-1]
        for i in self.renderlist:
            self.render_window.RemoveRenderer(i)
        self.render_window.AddRenderer(image)
        self.render_window.Render()
        self.iren = self.render_window.GetInteractor()

    def updateDataTable(self):
        filename = ''
        try:
            r = os.listdir(self.dirNameView)
            if self.tabWidget.currentIndex() == 0:
                for i in r:
                    if i.endswith('surface.csv'):
                        filename = self.dirNameView + '/' + i
                        break
                self.showDataTable(filename, self.tableView_3)
            elif self.tabWidget.currentIndex() == 1:
                for i in r:
                    if i.endswith('volume.csv'):
                        filename = self.dirNameView + '/' + i
                        break
                self.showDataTable(filename, self.tableView_2)
            elif self.tabWidget.currentIndex() == 2:
                for i in r:
                    if i.endswith('contact.csv'):
                        filename = self.dirNameView + '/' + i
                        break
                self.showDataTable(filename, self.tableView)
            elif self.tabWidget.currentIndex() == 3:
                if self.renderlist == []:
                    r = os.listdir(self.reconstructView)
                    self.maxNum = len(r)
                    self.Slider_3d.setMaximum(self.maxNum)
                    self.create3DImage()
                    self.set3DImage(1)
                    self.Label_idx.setText('{}/{}'.format('%03d' % 1, '%03d' % self.maxNum))

        except Exception:
            pass

    def runAll(self):
        config = {}
        try:
            # preprocess
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

            # segmentation
            config['para'] = {}
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

            # analysis
            config['para2'] = {}
            config['para2']['num_slice'] = int(self.LE_sliceNum_Ana.text())
            config['para2']['xy_resolution'] = float(self.LE_xyResolution_Ana.text())
            config['para2']['raw_folder'] = self.LE_rawFolder_Ana.text()
            en = []
            en.append(self.CB_embryoNames_Ana.currentText())
            config["para2"]["embryo_names"] = en
            config['para2']['project_folder'] = self.LE_projectFolder_Ana.text()
            config['para2']['first_run'] = False
            config['para2']["number_dictionary"] = self.LE_numberDict_Ana.text()
            config['para2']["lineage_file"] = self.LE_lineage_Ana.text()
            self.dirNameView = self.LE_projectFolder_Ana.text() + '/StatShape/' + self.CB_embryoNames_Ana.currentText()
            self.reconstructView = self.LE_projectFolder_Ana.text() + '/CellMembranePostseg/' + self.CB_embryoNames_Ana.currentText() + 'LabelUnified/'
            self.embryo = self.CB_embryoNames_Ana.currentText()
        except Exception:
            QMessageBox.warning(self, 'Error!', 'Please check your paras!')

        self.allthread = RunAllThread(config)
        self.allthread.signal.connect(self.ThreadCallback)
        self.allthread.process.connect(self.ProcessCallback)
        self.allthread.segmentation.connect(self.SegmentationCallback)
        self.allthread.analysis.connect(self.AnalysisCallback)
        self.allthread.start()

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
        fileName, fileType = QtWidgets.QFileDialog.getOpenFileName(self, 'Choose Lineage File',
                                                                   self.LE_rawFolder.text(), "CSV Files(*.csv)")
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
        self.PreprocessCall = False
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
                self.PreprocessCall = True
            elif func == 'Segmentation':
                self.SegmentationBar.setValue(100)
                self.SegmentationCall = True
            elif func == 'Analysis':
                self.AnalysisBar.setValue(100)
                self.AnalysisCall = True
            QMessageBox.information(self, func, func + ' success!')
        elif call == False:
            QMessageBox.warning(self, 'Error!', func + ' failed!')
        else:
            pass

    def ProcessCallback(self, func, current, max_time):
        self.label_Preprocess.setText(func + ':')
        self.PreprogressBar.setValue((current + 1) * 100 / max_time)

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
            model_name = re.findall(r'^.*.ckpt', fileName)
            self.LE_modelFile_Seg.setText(model_name[0])
        except Exception as e:
            QMessageBox.warning(self, 'Warning!', 'Please Choose Right Model!')

    def runSegmentation(self):
        config = {}
        try:
            config['para'] = {}
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
        self.SegmentationCall = False
        self.sthread = SegmentationThread(config)
        self.sthread.signal.connect(self.ThreadCallback)
        self.sthread.process.connect(self.SegmentationCallback)
        self.sthread.start()

    def SegmentationCallback(self, func, current, max_time):
        self.label_Segmentation.setText(func + ':')
        self.SegmentationBar.setValue((current + 1) * 100 / max_time)

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
        dirName = QtWidgets.QFileDialog.getExistingDirectory(self, 'Choose Project Folder', './')
        try:
            self.LE_projectFolder_Ana.setText(dirName)
        except Exception as e:
            QMessageBox.warning(self, 'Warning!', 'Please Choose Right Folder!')

    def chooseNumberDict_Ana(self):
        fileName, fileType = QtWidgets.QFileDialog.getOpenFileName(self, 'Choose Number Dictionary',
                                                                   './', "CSV Files(*.csv)")
        try:
            self.LE_numberDict_Ana.setText(fileName)
        except Exception as e:
            QMessageBox.warning(self, 'Warning!', 'Please Choose Right Folder!')

    def chooseLineageFile_Ana(self):
        fileName, fileType = QtWidgets.QFileDialog.getOpenFileName(self, 'Choose Lineage File', './',
                                                                   "CSV Files(*.csv)")
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
            self.dirNameView = self.LE_projectFolder_Ana.text() + '/StatShape/' + self.CB_embryoNames_Ana.currentText()
            self.reconstructView = self.LE_projectFolder_Ana.text() + '/CellMembranePostseg/' + self.CB_embryoNames_Ana.currentText() + 'LabelUnified/'
            self.embryo = self.CB_embryoNames_Ana.currentText()
        except Exception:
            QMessageBox.warning(self, 'Error!', 'Please check your paras!')
        self.AnalysisCall = False
        self.athread = AnalysisThread(config)
        self.athread.signal.connect(self.ThreadCallback)
        self.athread.process.connect(self.AnalysisCallback)
        self.athread.start()

    def AnalysisCallback(self, func, current, max_time):
        self.label_Analysis.setText(func + ':')
        self.AnalysisBar.setValue((current + 1) * 100 / max_time)

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

        self.PreprogressBar.reset()
        self.label_Preprocess.setText('')
        self.SegmentationBar.reset()
        self.label_Segmentation.setText('')
        self.AnalysisBar.reset()
        self.label_Analysis.setText('')
        self.tableView.reset()
        self.tableView_2.reset()
        self.tableView_3.reset()
        self.renderlist = []
        # if self.LE_rawFolder.text() != '':
        #     listdir = os.listdir(self.LE_rawFolder.text())
        #     listdir.sort()
        #     self.CB_embryoNames.addItems(listdir)
        # if self.LE_projectFolder_Seg.text() != '':
        #     listdir = os.listdir(self.LE_projectFolder_Seg.text())
        #     listdir.sort()
        #     self.CB_embryoNames_Seg.addItems(listdir)
        # if self.LE_rawFolder_Ana.text() != '':
        #     listdir = os.listdir(self.LE_rawFolder_Ana.text())
        #     listdir.sort()
        #     self.CB_embryoNames_Ana.addItems(listdir)

    def saveProject(self):
        # dirName = QtWidgets.QFileDialog.getExistingDirectory(self, 'Choose Save Folder', './')
        fileName, ok = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Project File', './', 'Project File(*.project)')
        try:
            # save all the paras
            with open(fileName, 'w', encoding='utf8', newline='') as fout:

                for i in self.findChildren(QLineEdit):
                    fout.write(i.objectName() + ':' + i.text() + '\n')
                for i in self.findChildren(QComboBox):
                    fout.write(i.objectName() + ':' + i.currentText() + '\n')
        except Exception as e:
            QMessageBox.warning(self, 'Warning!', 'Project Save Failed!')

    def loadProject(self):
        fileName, fileType = QtWidgets.QFileDialog.getOpenFileName(self, 'Choose Project File', './',
                                                                   "Project File(*.project)")

        try:
            self.PreprogressBar.reset()
            self.label_Preprocess.setText('')
            self.SegmentationBar.reset()
            self.label_Segmentation.setText('')
            self.AnalysisBar.reset()
            self.label_Analysis.setText('')
            self.tableView.reset()
            self.tableView_2.reset()
            self.tableView_3.reset()
            self.renderlist = []
            with open(fileName, 'r', encoding='utf8') as fr:
                r = fr.readlines()
                for i in r:
                    temp = i.split(':')
                    if len(temp) < 3:
                        temp[1] = temp[1].strip('\n')
                        if self.findChild(QLineEdit, temp[0]) is not None:
                            self.findChild(QLineEdit, temp[0]).setText(temp[1])
                        elif self.findChild(QComboBox, temp[0]) is not None:
                            self.findChild(QComboBox, temp[0]).setCurrentText(temp[1])
                    else:
                        temp_text = temp[1] + ':' + temp[2].strip('\n')
                        if self.findChild(QLineEdit, temp[0]) is not None:
                            self.findChild(QLineEdit, temp[0]).setText(temp_text)
                        elif self.findChild(QComboBox, temp[0]) is not None:
                            self.findChild(QComboBox, temp[0]).setCurrentText(temp_text)
            if self.LE_rawFolder.text() != '':
                listdir = os.listdir(self.LE_rawFolder.text())
                listdir.sort()
                self.CB_embryoNames.addItems(listdir)
            if self.LE_projectFolder_Seg.text() != '':
                listdir = os.listdir(self.LE_projectFolder_Seg.text())
                listdir.sort()
                self.CB_embryoNames_Seg.addItems(listdir)
            if self.LE_rawFolder_Ana.text() != '':
                listdir = os.listdir(self.LE_rawFolder_Ana.text())
                listdir.sort()
                self.CB_embryoNames_Ana.addItems(listdir)
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
                folder = self.LE_projectFolder.text()
            elif self.LE_projectFolder_Seg.text() != '':
                folder = self.LE_projectFolder_Seg.text()
            elif self.LE_projectFolder_Ana.text() != '':
                folder = self.LE_projectFolder_Ana.text()
            if platform.system() == 'Windows':
                os.startfile(folder)
            else:
                subprocess.call(['open', folder])
        except Exception:
            QMessageBox.warning(self, 'Warning!', 'Open Result Folder Failed!')

    def versionAbout(self):
        try:
            webbrowser.open('https://github.com/cao13jf/CShaperAPP.git')
        except Exception:
            pass

    def helpAbout(self):
        QMessageBox.information(self, 'About CShaper', 'CShaper APP is used to segment C. elegans embryo at single-cell level. Cell volume, cell surface and cell-cell contact surface are collected automatically in order to facilitate morphological researches on C. elegans.')



if __name__ == '__main__':
    freeze_support()
    app = QtWidgets.QApplication(sys.argv)
    win = MainForm()
    child = ChildDialog()
    child.resize(800,400)
    child.setFixedSize(800,400)
    win.actionHelp.triggered.connect(child.show)
    win.show()
    win.iren.Initialize()
    sys.exit(app.exec())
