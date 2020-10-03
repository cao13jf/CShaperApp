
from CShaper import Ui_MainWindow
from PyQt5.QtWidgets import (QApplication, QGridLayout, QGroupBox, QDialog,
                             QLabel, QSlider, QVBoxLayout, QMainWindow,
                             QMessageBox)
from PyQt5 import QtWidgets
from FuncThread import PreprocessThread, SegmentationThread, AnalysisThread
import warnings
from multiprocessing import freeze_support
import re
import sys
from ShapeUtil.data_structure import *
warnings.filterwarnings("ignore")


class MainForm(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainForm, self).__init__()
        self.setupUi(self)

        self.Function.currentChanged.connect(self.updateBlankInfo)
        # combine_slice.py
        self.Btn_rawFolder.clicked.connect(self.chooseRawFolder_Pre)
        self.Btn_projectFolder.clicked.connect(self.chooseProjectFolder_Pre)
        self.Btn_lineageFile.clicked.connect(self.chooseLineageFile_Pre)
        self.Btn_runPreprocess.clicked.connect(self.runPreprocess)
        self.Btn_numberDict.clicked.connect(self.chooseNumberDict)

        # test_edt.py
        self.Btn_projectFolder_Seg.clicked.connect(self.chooseProjectFolder_Seg)
        self.Btn_modelFile_Seg.clicked.connect(self.chooseModelFile_Seg)
        self.Btn_runSegmentation.clicked.connect(self.runSegmentation)

        # shape_analysis.py
        self.Btn_runAnalysis.clicked.connect(self.runAnalysis)
        self.Btn_numberDict_Ana.clicked.connect(self.chooseNumberDict_Ana)
        self.Btn_rawFolder_Ana.clicked.connect(self.chooseRawFolder_Ana)
        self.Btn_projectFolder_Ana.clicked.connect(self.chooseProjectFolder_Ana)
        self.Btn_lineageFile_Ana.clicked.connect(self.chooseLineageFile_Ana)

        # run all
        self.Btn_runAll.clicked.connect(self.runAll)

    def updateBlankInfo(self):
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
        self.thread = PreprocessThread(config)
        self.thread.signal.connect(self.ThreadCallback)
        self.thread.start()

    def ThreadCallback(self, call, func):

        if call:
            QMessageBox.information(self, func, func+' success!')
        else:
            QMessageBox.warning(self, 'Error!', func+' failed!')
        # print(call)

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
        self.thread = SegmentationThread(config)
        self.thread.signal.connect(self.ThreadCallback)
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
        except Exception:
            QMessageBox.warning(self, 'Error!', 'Please check your paras!')
        self.call = False
        self.thread = AnalysisThread(config)
        self.thread.signal.connect(self.ThreadCallback)
        self.thread.start()


if __name__ == '__main__':
    freeze_support()
    app = QtWidgets.QApplication(sys.argv)
    win = MainForm()
    win.show()
    sys.exit(app.exec())