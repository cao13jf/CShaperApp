
from CShaper import Ui_MainWindow

import sys
import os

from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import (QApplication, QGridLayout, QGroupBox, QDialog,
                             QLabel, QSlider, QVBoxLayout, QMainWindow,
                             QMessageBox)
from PyQt5 import QtWidgets
from FuncThread import PreprocessThread, SegmentationThread, AnalysisThread
import shutil
from PIL import Image
from skimage.measure import marching_cubes_lewiner, mesh_surface_area

import tensorflow as tf
from multiprocessing import freeze_support
from ShapeUtil.draw_lib import *
from ShapeUtil.data_structure import *

from Util.data_loader import *
from Util.train_test_func import *
from Util.segmentation_post_process import *
from Util.train_test_func import prediction_fusion
from train import NetFactory
import gc
import re
import sys
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy import ndimage
import multiprocessing as mp
from skimage import morphology
from skimage.measure import marching_cubes_lewiner, mesh_surface_area
from ShapeUtil.draw_lib import *
from ShapeUtil.data_structure import *
from Util.post_lib import check_folder_exist
from Util.parse_config import parse_config
from Util.segmentation_post_process import save_nii
import warnings


class MainForm(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainForm, self).__init__()
        self.setupUi(self)

        self.Function.currentChanged.connect(self.updateBlankInfo)
        # combine_slice.py
        self.Btn_rawFolder.clicked.connect(self.chooseRawFolder_Pre)
        self.Btn_stackFolder.clicked.connect(self.chooseStackFolder_Pre)
        self.Btn_lineageFile.clicked.connect(self.chooseLineageFile_Pre)
        self.Btn_runPreprocess.clicked.connect(self.runPreprocess)
        self.Btn_shapeFile.clicked.connect(self.chooseShapeFile)

        # test_edt.py
        self.Btn_stackFolder_Seg.clicked.connect(self.chooseStackFolder_Seg)
        self.Btn_saveFolder_Seg.clicked.connect(self.chooseSaveFolder_Seg)
        self.Btn_modelFile_Seg.clicked.connect(self.chooseModelFile_Seg)
        self.Btn_runSegmentation.clicked.connect(self.runSegmentation)

        # shape_analysis.py
        self.Btn_runAnalysis.clicked.connect(self.runAnalysis)
        self.Btn_shapeUtil_Ana.clicked.connect(self.chooseShapeUtil)
        self.Btn_rawFolder_Ana.clicked.connect(self.chooseRawFolder_Ana)
        self.Btn_stackFolder_Ana.clicked.connect(self.chooseStackFolder_Ana)
        self.Btn_saveFolder_Ana.clicked.connect(self.chooseSaveFolder_Ana)

        # run all
        self.Btn_runAll.clicked.connect(self.runAll)

        # self.LE_sliceNum_Ana.setText('68')
        # self.LE_xyResolution_Ana.setText('0.09')
        # self.LE_rawFolder_Ana.setText(r'E:\CityU\CShaperAPP\Data\MembRaw')
        # self.LE_saveFolder_Ana.setText(r'E:\CityU\CShaperAPP\ResultCell')
        # self.LE_stackFolder_Ana.setText(r'E:\CityU\CShaperAPP\Data\MembTest')
        # self.LE_shapeUtil_Ana.setText(r'E:\CityU\CShaperAPP\ShapeUtil')

    def updateBlankInfo(self):
        if self.Function.currentIndex() == 1:
            if self.LE_stackFolder.text() != '':
                self.LE_stackFolder_Seg.setText(self.LE_stackFolder.text())
                try:
                    self.CB_embryoNames_Seg.clear()
                    listdir = os.listdir(self.LE_stackFolder.text())
                    listdir.sort()
                    self.CB_embryoNames_Seg.addItems(listdir)
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
            if self.LE_stackFolder_Seg.text() != '':
                self.LE_stackFolder_Ana.setText(self.LE_stackFolder_Seg.text())
            elif self.LE_stackFolder.text() != '':
                self.LE_stackFolder_Ana.setText(self.LE_stackFolder.text())
        # print( self.Function.currentIndex())


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

    def chooseStackFolder_Pre(self):
        dirName = QtWidgets.QFileDialog.getExistingDirectory(self, 'Choose Stack Folder', './')
        try:
            self.LE_stackFolder.setText(dirName)
        except Exception as e:
            QMessageBox.warning(self, 'Warning!', 'Please Choose Right Folder!')

    def chooseLineageFile_Pre(self):
        fileName, fileType = QtWidgets.QFileDialog.getOpenFileName(self, 'Choose Lineage File', self.LE_rawFolder.text(), "CSV Files(*.csv)")
        try:
            self.LE_lineage.setText(fileName)
        except Exception as e:
            QMessageBox.warning(self, 'Warning!', 'Please Choose Right Folder!')
    def chooseShapeFile(self):
        fileName, fileType = QtWidgets.QFileDialog.getOpenFileName(self, 'Choose Lineage File',
                                                                   './', "TXT Files(*.txt)")
        try:
            self.LE_shapeFile.setText(fileName)
        except Exception as e:
            QMessageBox.warning(self, 'Warning!', 'Please Choose Right Folder!')

    def runPreprocess(self):
        config = {}
        config['num_slice'] = int(self.LE_sliceNum.text())
        en = []
        en.append(self.CB_embryoNames.currentText())
        config["embryo_names"] = en
        config["max_time"] = int(self.LE_maxTime.text())
        config["xy_resolution"] = float(self.LE_xyResolution.text())
        config["z_resolution"] = float(self.LE_zResolution.text())
        config["reduce_ratio"] = float(self.LE_reduceRatio.text())
        config["raw_folder"] = self.LE_rawFolder.text()
        config["stack_folder"] = self.LE_stackFolder.text()
        config["lineage_file"] = self.LE_lineage.text()
        config["shape_file"] = self.LE_shapeFile.text()

        self.LE_maxTime_Seg.setText(self.LE_maxTime.text())
        self.LE_sliceNum_Ana.setText(self.LE_sliceNum.text())
        self.thread = PreprocessThread(config)
        self.thread.signal.connect(self.PreprocessCallback)
        self.thread.start()
        # print(self.thread.finished())
        # try:
        #     combine_slices(config)
        # except Exception as e:
        #     QMessageBox.warning(self, 'Errors', 'Parameters error, please check your paras!')
    def PreprocessCallback(self, call):
        self.call = call
        if self.call :
            QMessageBox.information(self, 'Mission success')
        else:
            QMessageBox.warning(self, 'Please check your paras!')
        # print(call)

    def chooseStackFolder_Seg(self):
        dirName = QtWidgets.QFileDialog.getExistingDirectory(self, 'Choose Stack Folder', './')
        try:
            self.LE_stackFolder_Seg.setText(dirName)
            self.CB_embryoNames_Seg.clear()
            listdir = os.listdir(dirName)
            listdir.sort()
            self.CB_embryoNames_Seg.addItems(listdir)
        except Exception as e:
            QMessageBox.warning(self, 'Warning!', 'Please Choose Right Folder!')

    def chooseSaveFolder_Seg(self):
        dirName = QtWidgets.QFileDialog.getExistingDirectory(self, 'Choose Raw Folder', './')
        try:
            self.LE_saveFolder_Seg.setText(dirName)
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
        config['para']={}
        config["para"]["stack_folder"] = self.LE_stackFolder_Seg.text()
        en = []
        en.append(self.CB_embryoNames_Seg.currentText())
        config["para"]["embryo_names"] = en
        config["para"]["max_time"] = int(self.LE_maxTime_Seg.text())
        config["para"]["save_folder"] = self.LE_saveFolder_Seg.text()
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
        config["data"]["data_root"] = config["para"]["stack_folder"]
        config["data"]["data_names"] = config["para"]["embryo_names"]
        config["data"]["max_time"] = config["para"]["max_time"]
        config["data"]["save_folder"] = config["para"]["save_folder"]
        config["data"]["with_ground_truth"] = False
        config["data"]["label_edt_transform"] = True
        config["data"]["valid_edt_width"] = 30
        config["data"]["label_edt_discrete"] = True
        config["data"]["edt_discrete_num"] = 16
        config["data"]["save_folder"] = os.path.join(config["data"]["save_folder"], "BinaryMemb")

        config["network"] = {}
        config["network"]["net_type"] = "DMapNet"
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
        config["segdata"]["membseg_path"] = os.path.dirname(config["data"]["save_folder"])
        config["segdata"]["nucleus_data_root"] = config["data"]["data_root"]

        config["debug"] = {}
        config["debug"]["debug_mode"] = False
        config["debug"]["save_anisotropic"] = False
        config["debug"]["save_graph_model"] = False
        config["debug"]["save_init_watershed"] = False
        config["debug"]["save_merged_seg"] = False
        config["debug"]["save_cell_nomemb"] = False

        self.thread = SegmentationThread(config)
        self.thread.signal.connect(self.PreprocessCallback)
        self.thread.start()

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

    def chooseStackFolder_Ana(self):
        dirName = QtWidgets.QFileDialog.getExistingDirectory(self, 'Choose Raw Folder', './')
        try:
            self.LE_stackFolder_Ana.setText(dirName)
        except Exception as e:
            QMessageBox.warning(self, 'Warning!', 'Please Choose Right Folder!')

    def chooseSaveFolder_Ana(self):
        dirName = QtWidgets.QFileDialog.getExistingDirectory(self, 'Choose Raw Folder', './')
        try:
            self.LE_saveFolder_Ana.setText(dirName)
        except Exception as e:
            QMessageBox.warning(self, 'Warning!', 'Please Choose Right Folder!')

    def chooseShapeUtil(self):
        dirName = QtWidgets.QFileDialog.getExistingDirectory(self, 'Choose Raw Folder', './')
        try:
            self.LE_shapeUtil_Ana.setText(dirName)
        except Exception as e:
            QMessageBox.warning(self, 'Warning!', 'Please Choose Right Folder!')

    def runAnalysis(self):

        config = {}
        config['para'] = {}
        config['para']['num_slice'] = int(self.LE_sliceNum_Ana.text())
        config['para']['xy_resolution'] = float(self.LE_xyResolution_Ana.text())
        config['para']['raw_folder'] = self.LE_rawFolder_Ana.text()
        config['para']['save_folder'] = self.LE_saveFolder_Ana.text()
        en = []
        en.append(self.CB_embryoNames_Ana.currentText())
        config["para"]["embryo_names"] = en
        config['para']['stack_folder'] = self.LE_stackFolder_Ana.text()
        config['para']['first_run'] = False
        config['para']["label_dict"] = self.LE_shapeUtil_Ana.text()

        self.thread = AnalysisThread(config)
        self.thread.signal.connect(self.PreprocessCallback)
        self.thread.start()





if __name__ == '__main__':
    freeze_support()
    app = QtWidgets.QApplication(sys.argv)
    win = MainForm()
    win.show()
    sys.exit(app.exec())