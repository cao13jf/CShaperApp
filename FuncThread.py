
from PyQt5.QtCore import pyqtSignal, QThread
from Util.preprocess_lib import *
from shape_analysis import *
from test_edt import *

class PreprocessThread(QThread):
    signal = pyqtSignal(bool, str)
    process = pyqtSignal(str, int, int)
    def __init__(self, config={}):
        self.config = config
        super(PreprocessThread, self).__init__()

    def __del__(self):
        self.wait()

    def run(self):
        # try:
        combine_slices(self.process, self.config)
        self.signal.emit(True, 'Preprocess')
        # except Exception:
        #     self.signal.emit(False, 'Preprocess')


class SegmentationThread(QThread):
    signal = pyqtSignal(bool, str)
    process = pyqtSignal(str, int, int)
    def __init__(self, config={}):
        self.config = config
        super(SegmentationThread, self).__init__()

    def __del__(self):
        self.wait()

    def run(self):
        try:
            test(self.process, self.config)
            self.signal.emit(True, 'Segmentation')
        except Exception:
            self.signal.emit(False, 'Segmentation')


class AnalysisThread(QThread):
    signal = pyqtSignal(bool, str)
    process = pyqtSignal(str, int, int)
    def __init__(self, config={}):
        self.config = config
        super(AnalysisThread, self).__init__()

    def __del__(self):
        self.wait()

    def run(self):
        try:
            # config = self.config
            para_config = self.config['para']
            # print(para_config)
            para_config["data_folder"] = os.path.join(para_config["project_folder"], "RawStack")
            para_config["save_nucleus_folder"] = os.path.join(para_config["project_folder"], "NucleusLoc")
            para_config["seg_folder"] = os.path.join(para_config["project_folder"], "CellMembranePostseg")
            para_config["stat_folder"] = os.path.join(para_config["project_folder"], "StatShape")
            para_config["delete_tem_file"] = False

            if not os.path.isdir(para_config['stat_folder']):
                os.makedirs(para_config['stat_folder'])
            # Get the size of the figure
            example_embryo_folder = os.path.join(para_config["raw_folder"], para_config["embryo_names"][0], "tif")
            example_img_file = glob.glob(os.path.join(example_embryo_folder, "*.tif"))
            raw_size = [para_config["num_slice"]] + list(np.asarray(Image.open(example_img_file[0])).shape)
            para_config["image_size"] = [raw_size[0], raw_size[2], raw_size[1]]

            para_config["embryo_name"] = para_config["embryo_names"][0]
            para_config["acetree_file"] = para_config["lineage_file"]
            if not os.path.isdir(os.path.join(para_config['save_nucleus_folder'], para_config['embryo_name'])):
                os.makedirs(os.path.join(para_config['save_nucleus_folder'], para_config['embryo_name']))
            else:
                shutil.rmtree(os.path.join(para_config['save_nucleus_folder'], para_config['embryo_name']))
                os.makedirs(os.path.join(para_config['save_nucleus_folder'], para_config['embryo_name']))
            run_shape_analysis(self.process, para_config)

            self.signal.emit(True, 'Analysis')
        except Exception:
            self.signal.emit(False, 'Analysis')
