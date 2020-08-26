
import os
import glob
import pickle
import warnings
import shutil
import numpy as np
import nibabel as nib
from tqdm import tqdm
import pandas as pd
from csv import reader, writer

from Util.data_process import load_3d_volume_as_array

# ***********************************************
# functions
# ***********************************************
def test_folder(folder_name):
    if "." in folder_name[1:]:
        folder_name = os.path.dirname(folder_name)
    if not os.path.isdir(folder_name):
        os.makedirs(folder_name)

def transpose_csv(source_file, target_file):
    with open(source_file) as f, open(target_file, 'w') as fw:
        writer(fw, delimiter=',').writerows(zip(*reader(f, delimiter=',')))


RENAME_FLAG = True
CELLSPAN_FLAG = False
TP_CELLS_FLAGE = False
LOST_CELL = False
NEIGHBOR_FLAG = False
COPY_FILE = False

embryo_names = ["200113plc1p2"]

# =================================================
# write header (https://brainder.org/2012/09/23/the-nifti-file-format/)
# =================================================
if RENAME_FLAG:
    res_embryos = {0.25: ['170704plc1p1',
                  '181210plc1p1',
                  '181210plc1p2',
                  '181210plc1p3',
                  '200309plc1p1',
                  '200309plc1p2',
                  '200309plc1p3',
                  '200311plc1p1',
                  '200311plc1p3',
                  '200312plc1p2',
                  '200314plc1p1',
                  '200314plc1p2',
                  '200314plc1p3',
                  '200315plc1p2',
                  '200315plc1p3',
                  '200316plc1p1',
                  '200316plc1p2',
                  '200316plc1p3',
                  '200310plc1p2',
                  '200311plc1p2',
                  '200312plc1p1',
                  '200312plc1p3',
                  '200315plc1p1',
                  '200113plc1p2',
                  '200113plc1p3',
                  '200109plc1p1'],
           0.18: ['200710hmr1plc1p1',
                  '200710hmr1plc1p2',
                  '200710hmr1plc1p3'],
    }
    # "/home/jeff/ProjectCode/LearningCell/MembProjectCode/dataset/test/200113plc1p2"
    data_folder = "/home/jeff/ProjectCode/LearningCell/MembProjectCode/output/200113plc1p2LabelUnified"
    data_files = glob.glob(os.path.join(data_folder, "**/*.nii.gz"), recursive=True)
    for data_file in tqdm(data_files, desc="Adding header"):
        img = nib.load(data_file)
        img.header.set_xyzt_units(xyz=3, t=8)
        res_flag = False
        for res, embryos in res_embryos.items():
            if any([embryo in data_file for embryo in embryos]):
                res_flag = True
                img.header["pixdim"] = [1.0, res, res, res, 0., 0., 0., 0.]
                nib.save(img, data_file, )
                break
        if not res_flag:
            warnings.warn("No resolution for {}!".format(data_file.split("/")[-1]))


# ==============================================================================
# generate data for GUI
# ==============================================================================
with open("/home/jeff/ProjectCode/LearningCell/MembProjectCode/dataset/number_dictionary.txt", 'rb') as f:
    number_dict = pickle.load(f)

# =================== save cell life span ======================================
if CELLSPAN_FLAG:
    for embryo_name in embryo_names:
        write_file = "./Tem/GUIData/{}/{}_lifescycle.csv".format(embryo_name, embryo_name)
        test_folder(write_file)

        with open("/home/jeff/ProjectCode/LearningCell/MembProjectCode/ResultCell/StatShape/{}_time_tree.txt".format(embryo_name), 'rb') as f:
            time_tree = pickle.load(f)
        all_cells = list(time_tree.nodes)
        i = 0
        for i, one_cell in enumerate(tqdm(all_cells, desc="Life span {}".format(embryo_name))):
            times = time_tree.get_node(one_cell).data
            if times is None:
                continue
            times = [number_dict[one_cell]] + times.get_time()
            write_string = ",".join([str(x) for x in times]) + "\n"
            if i == 0:
                i = 1
                with open(write_file, "w") as f:
                    f.write(write_string)
            else:
                with open(write_file, "a") as f:
                    f.write(write_string)


# # ======================== generate TP Cells =============================================
if TP_CELLS_FLAGE:
    for embryo_name in embryo_names:
        save_folder = "./Tem/GUIData/{}/TPCell".format(embryo_name, embryo_name)
        test_folder(save_folder)

        folder_name = os.path.join("/home/jeff/ProjectCode/LearningCell/MembProjectCode/output", embryo_name + "LabelUnified")
        file_list = glob.glob(os.path.join(folder_name, "*.nii.gz"))
        file_list.sort()
        for file_name in tqdm(file_list, desc="TP Cells {}".format(embryo_name)):
            seg = load_3d_volume_as_array(file_name)
            cell_labels = np.unique(seg).tolist()
            cell_labels.sort()
            cell_labels.remove(0)
            cell_string = ",".join(([str(cell_label) for cell_label in cell_labels]))

            base_name = os.path.basename(file_name)
            save_file = os.path.join(save_folder, "_".join(base_name.split("_")[:2]+["cells.txt"]))
            with open(save_file, "w") as f:
                f.write(cell_string+"\n")

# # get lost cells
if LOST_CELL:
    for embryo_name in embryo_names:
        seg_folder = os.path.join("/home/jeff/ProjectCode/LearningCell/MembProjectCode/output", embryo_name + "LabelUnified")
        nucleus_folder = os.path.join("/home/jeff/ProjectCode/LearningCell/MembProjectCode/dataset/test", embryo_name, "SegNuc")
        seg_files = glob.glob(os.path.join(seg_folder, "*.nii.gz"))
        nucleus_files = glob.glob(os.path.join(nucleus_folder, "*.nii.gz"))
        seg_files.sort()
        nucleus_files.sort()

        save_folder = "./Tem/GUIData/{}/LostCell".format(embryo_name, embryo_name)
        test_folder(save_folder)

        for i, seg_file in enumerate(tqdm(seg_files, desc="Lost cells {}".format(embryo_name))):
            nucleus_file = nucleus_files[i]
            seg = load_3d_volume_as_array(seg_file)
            nucleus = load_3d_volume_as_array(nucleus_file)

            lost_cells = np.unique(nucleus[seg == 0]).tolist()
            lost_cells.remove(0)
            cell_string = ",".join(([str(cell_label) for cell_label in lost_cells]))

            base_name = os.path.basename(seg_file)
            save_file = os.path.join(save_folder, "_".join(base_name.split("_")[:2]+["lostCell.txt"]))
            with open(save_file, "w") as f:
                f.write(cell_string + "\n")

# # get neighboer
if NEIGHBOR_FLAG:
    for embryo_name in embryo_names:
        file_list = glob.glob(os.path.join('./ShapeUtil/TemCellGraph', embryo_name, embryo_name + '_T*.txt'))
        file_list = [file for file in file_list if "nucLoc" not in file]
        for file_name in tqdm(file_list, desc="Getting neighbors {}".format(embryo_name)):
            with open(file_name, 'rb') as f:
                cell_graph = pickle.load(f)

            tp = os.path.basename(file_name).split("_")[1][1:-4].zfill(3)
            base_name = "_".join([embryo_name, tp])
            save_file = os.path.join("./Tem/GUIData", embryo_name, "GuiNeighbor", base_name+"_guiNeighbor.txt")
            test_folder(save_file)

            for i, cell_name in enumerate(cell_graph.nodes()):
                neighbor_cells = list(cell_graph.neighbors(cell_name))
                neighbor_labels = [str(number_dict[name]) for name in neighbor_cells]
                cell_label = str(number_dict[cell_name])
                label_str = ",".join(([cell_label] + neighbor_labels))  # first one master cell
                if i == 0:
                    with open(save_file, "w") as f:
                        f.write(label_str+"\n")
                else:
                    with open(save_file, "a") as f:
                        f.write(label_str+"\n")

# ================== copy files ==============================
if COPY_FILE:
    # volume
    for embryo_name in embryo_names:
        shutil.copyfile(os.path.join("./ShapeUtil/RobustStat", embryo_name + "_surface.csv"),
               os.path.join("./Tem/GUIData", embryo_name, embryo_name + "_surface.csv"))
        # surface
        shutil.copyfile(os.path.join("./ShapeUtil/RobustStat", embryo_name + "_volume.csv"),
               os.path.join("./Tem/GUIData", embryo_name, embryo_name + "_volume.csv"))
        # contact (with transpose)
        transpose_csv(os.path.join("/home/jeff/ProjectCode/LearningCell/MembProjectCode/ResultCell/StatShape", embryo_name + "_Stat.csv"),
               os.path.join("./Tem/GUIData", embryo_name, embryo_name + "_Stat.csv"))

    # shutil.copyfile("./ShapeUtil/name_dictionary.csv", "./Tem/GUIData/name_dictionary.csv")