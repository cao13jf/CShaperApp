
from PyQt5.QtCore import pyqtSignal, QThread
from Util.preprocess_lib import combine_slices
import tensorflow as tf
from Util.data_loader import *
from Util.train_test_func import *
from Util.segmentation_post_process import *
from Util.train_test_func import prediction_fusion
from train import NetFactory
import gc

import shutil
import warnings
from PIL import Image
from tqdm import tqdm
from scipy import ndimage
import multiprocessing as mp
from skimage.measure import marching_cubes_lewiner, mesh_surface_area
from ShapeUtil.draw_lib import *
from ShapeUtil.data_structure import *
from Util.post_lib import check_folder_exist
from Util.segmentation_post_process import save_nii


class PreprocessThread(QThread):
    signal = pyqtSignal(bool, str)

    def __init__(self, config={}):
        self.config = config
        super(PreprocessThread, self).__init__()

    def __del__(self):
        self.wait()

    def run(self):
        try:
            combine_slices(self.config)
            self.signal.emit(True, 'Preprocess')
        except Exception:
            self.signal.emit(False, 'Preprocess')


class SegmentationThread(QThread):
    signal = pyqtSignal(bool, str)

    def __init__(self, config={}):
        self.config = config
        super(SegmentationThread, self).__init__()

    def __del__(self):
        self.wait()

    def run(self):
        try:
            config = self.config
            config_data = config['data']
            config_net = config.get('network', None)
            config_test = config['testing']
            batch_size = config_test.get('batch_size', 5)
            label_edt_discrete = config_data.get('label_edt_discrete', False)
            if not config_test['only_post_process']:
                net_type = config_net['net_type']
                net_name = config_net['net_name']
                data_shape = config_net['data_shape']
                label_shape = config_net['label_shape']
                class_num = config_data.get('edt_discrete_num', 16)

                # ==============================================================
                #               2, Construct computation graph
                # ==============================================================
                full_data_shape = [None] + data_shape
                input_x = tf.placeholder(tf.float32, shape=full_data_shape)
                net_class = NetFactory.create(net_type)
                net = net_class(num_classes=class_num, w_regularizer=None, b_regularizer=None, name=net_name)
                net.set_params(config_net)

                net_output = net(input_x, is_training=True)

                # ==============================================================
                #               3, Data loader
                # ==============================================================
                dataloader = DataLoader(config_data)
                dataloader.load_data()
                [temp_imgs, img_names, emrbyo_name, temp_bbox, temp_size] = dataloader.get_image_data_with_name(0)

                # For axial direction
                temp_img_axial = transpose_volumes(temp_imgs, slice_direction='axial')
                [D, H, W] = temp_img_axial.shape
                Hx = max(int((H + 3) / 4) * 4, data_shape[1])
                Wx = max(int((W + 3) / 4) * 4, data_shape[2])
                data_slice = data_shape[0]
                full_data_shape = [None, data_slice, Hx, Wx, data_shape[-1]]
                x_axial = tf.placeholder(tf.float32, full_data_shape)
                predicty_axial = net(x_axial, is_training=True)
                proby_axial = tf.nn.softmax(predicty_axial)

                # For sagittal direction
                temp_imgs = transpose_volumes(temp_imgs, slice_direction='sagittal')
                [D, H, W] = temp_imgs.shape
                Hx = max(int((H + 3) / 4) * 4, data_shape[1])
                Wx = max(int((W + 3) / 4) * 4, data_shape[2])
                data_slice = data_shape[0]
                full_data_shape = [None, data_slice, Hx, Wx, data_shape[-1]]
                x_sagittal = tf.placeholder(tf.float32, full_data_shape)
                predicty_sagittal = net(x_sagittal, is_training=True)
                proby_sagittal = tf.nn.softmax(predicty_sagittal)

                # ==============================================================
                #               4, Start prediction
                # ==============================================================
                all_vars = tf.global_variables()
                sess = tf.InteractiveSession()
                sess.run(tf.global_variables_initializer())
                net1_vars = [x for x in all_vars if x.name[0:len(net_name) + 1] == net_name + '/']
                saver = tf.train.Saver(net1_vars)
                saver.restore(sess, config_net['model_file'])
                sess.graph.finalize()
                slice_direction = config_test.get('slice_direction', 'axial')
                save_folder = config_data['save_folder']
                if not os.path.isdir(save_folder):
                    os.makedirs(save_folder)  # If the target folder doesn't exist, create a new one
                test_time = []
                data_number = config_data.get('max_time', 100) * len(config_data["data_names"])
                for i in tqdm(range(0, data_number), desc='Segmenting Data:'):
                    [temp_imgs, img_names, emrbyo_name, temp_bbox, temp_size] = dataloader.get_image_data_with_name(i)
                    temp_img_sagittal = transpose_volumes(temp_imgs, slice_direction)
                    if (slice_direction == 'sagittal'):
                        tem_box = temp_bbox.copy()
                        temp_bbox = [[a[2], a[0], a[1]] for a in tem_box]
                        temp_size = (temp_size[2], temp_size[0], temp_size[1])
                    if (slice_direction == 'coronal'):
                        tem_box = temp_bbox.copy()
                        temp_bbox = [[a[1], a[0], a[2]] for a in tem_box]
                        temp_size = (temp_size[1], temp_size[0], temp_size[2])
                    t0 = time.time()
                    prob_sagittal = predict_full_volume(temp_img_sagittal, data_shape[:-1], label_shape[:-1],
                                                        data_shape[-1],
                                                        class_num, batch_size, sess, proby_sagittal, x_sagittal)
                    # Combine results from two different directions. In fusion stage, refinement is based on sagittal direction
                    if config_test.get('direction_fusion', False):
                        temp_img_axial = transpose_volumes(temp_imgs, slice_direction='axial')
                        prob_axial = predict_full_volume(temp_img_axial, data_shape[:-1], label_shape[:-1],
                                                         data_shape[-1],
                                                         class_num, batch_size, sess, proby_axial, x_axial)

                    # If the prediction is one-hot tensor, use np.argmax to get the indices of the maximumm. That indice is used as label
                    if (label_edt_discrete):  # If we hope to predict discrete EDT map, argmax on the indices depth
                        if config_test.get('direction_fusion', False):
                            prob_axial = np.transpose(prob_axial, [2, 0, 1, 3])
                            pred_fusion = prediction_fusion(prob_axial, prob_sagittal)
                        else:
                            pred_fusion = (np.argmax(prob_sagittal, axis=-1)).astype(np.uint16)
                        pred = delete_isolate_labels(pred_fusion)
                    else:
                        pred = prob_sagittal  # Regression prediction provides the MAP directly.

                    out_label = post_process_on_edt(pred).astype(np.int16)

                    # ==============================================================
                    #               save prediction results as *.nii.gz
                    # ==============================================================
                    test_time.append(time.time() - t0)
                    final_label = np.zeros(temp_size, out_label.dtype)
                    final_label = set_crop_to_volume(final_label, temp_bbox[0], temp_bbox[1], out_label)
                    final_label = transpose_volumes_reverse(final_label, slice_direction)
                    save_file = os.path.join(save_folder, emrbyo_name,
                                             emrbyo_name + "_" + img_names.split(".")[0].split("_")[
                                                 1] + "_segMemb.nii.gz")
                    save_array_as_nifty_volume(final_label,
                                               save_file)  # os.path.join(save_folder, one_embryo, one_embryo + "_" + tp_str.zfill(3) + "_cell.nii.gz")

                    print(save_file)
                test_time = np.asarray(test_time)
                print('test time', test_time.mean())
                sess.close()

                del sess, dataloader, net
                if __name__ != '__main__':
                    gc.collect()

            # ==============================================================
            #               Post processing (binary membrane --> isolated SegCell)
            # ==============================================================
            if config_test.get('post_process', False):
                config_post = {}
                config_post['segdata'] = config['segdata']
                config_post['segdata']['embryos'] = config_data['data_names']
                config_post['debug'] = config['debug']
                config_post['result'] = {}

                config_post['segdata']['membseg_path'] = config_data['save_folder']
                config_post['result']['postseg_folder'] = config_data['save_folder'] + 'Postseg'
                config_post['result']['nucleus_filter'] = config_test['nucleus_filter']
                config_post['result']['nucleus_as_seed'] = config_test['nucleus_as_seed']
                config_post['segdata']['max_time'] = config["para"]["max_time"]
                if not os.path.isdir(config_post['result']['postseg_folder']):
                    os.makedirs(config_post['result']['postseg_folder'])
                post_process(config_post)
                self.signal.emit(True, 'Segmentation')
        except Exception:
            self.signal.emit(False, 'Segmentation')


class AnalysisThread(QThread):
    signal = pyqtSignal(bool, str)

    def __init__(self, config={}):
        self.config = config
        super(AnalysisThread, self).__init__()

    def __del__(self):
        self.wait()

    def run(self):
        # try:
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
            # print(para_config)
            # embryo_names = para_config["embryo_names"]

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
            run_shape_analysis(para_config)

            self.signal.emit(True, 'Analysis')
        # except Exception:
        #     self.signal.emit(False, 'Analysis')


warnings.filterwarnings("ignore")
stat_embryo = None  # Global embryo shape information
max_time = None
cell_tree = None

def init(l):  # used for parallel computing
    global file_lock
    file_lock = l


def run_shape_analysis(config):
    '''
    Extract the cell tree structure from the aceTree file
    :param acetree_file:  file name of the embryo acetree file
    :param max_time:  the maximum time point in the tree.
    :return :
    '''
    global max_time
    global cell_tree
    ## construct lineage tree whose nodes contain the time points that cell exist (based on nucleus).
    acetree_file = config['acetree_file']
    cell_tree, max_time = construct_celltree(acetree_file, config)
    save_file_name = os.path.join(config['stat_folder'], config['embryo_name']+'_time_tree.txt')

    with open(save_file_name, 'wb') as f:
        pickle.dump(cell_tree, f)
    ## Parallel computing for the cell relation graph
    if not os.path.isdir(os.path.join(config["project_folder"], 'TemCellGraph')):
        os.makedirs(os.path.join(config["project_folder"], 'TemCellGraph'))

    pd_number = pd.read_csv(config["number_dictionary"], names=["name", "label"])
    number_dict = pd.Series(pd_number.label.values, index=pd_number.name).to_dict()

    # ========================================================
    #       sementing TPs in a parallel way
    # ========================================================
    file_lock = mp.Lock()  # |-----> for change treelib files
    print(file_lock, mp.cpu_count(), init)
    mpPool = mp.Pool(mp.cpu_count()-1, initializer=init, initargs=(file_lock,))  # TODO: change `mp.cpu_count()-1` --> `2`
    configs = []
    config["cell_tree"] = cell_tree
    for itime in tqdm(range(1, max_time+1), desc="Compose configs"):
        config['time_point'] = itime
        configs.append(config.copy())
    #     cell_graph_network(file_lock, config)
    embryo_name = config["embryo_names"][0]
    for _ in tqdm(mpPool.imap_unordered(cell_graph_network, configs), total=len(configs), desc="Naming {} segmentations".format(embryo_name)):
        pass


    # ========================================================
    #       Combine previous TPs
    # ========================================================
    # ## In order to make use of parallel computing, the global vairable stat_embryo cannot be shared between different processor,
    # #  so we need to store all one-embryo reults as temporary files, which will be assembled finally. After that, these temporary
    # #  Data can be deleted.
    construct_stat_embryo(cell_tree, max_time)  # initilize the shape matrix which is use to store the shape series information
    for itime in tqdm(range(1, max_time+1), desc='assembling {} result'.format(embryo_name)):
        file_name = os.path.join(config["project_folder"], 'TemCellGraph', config['embryo_name'], config['embryo_name']+'_T'+str(itime)+'.txt')
        with open(file_name, 'rb') as f:
            cell_graph = pickle.load(f)
            stat_embryo = assemble_result(cell_graph, itime, number_dict)
    if config['delete_tem_file']:  # If need to delete temporary files.
        shutil.rmtree(os.path.join(config["project_folder"], 'TemCellGraph'))
    # save statistical embryonic files
    # delete columns with all zeros for efficiency
    stat_embryo = stat_embryo.loc[:, ((stat_embryo != 0)&(~np.isnan(stat_embryo))).any(axis=0)]
    save_file_name = os.path.join(config['stat_folder'], config['embryo_name']+'_stat.txt')
    save_file_name_csv = os.path.join(config['stat_folder'], config['embryo_name']+'_stat.csv')
    if not os.path.isdir(config['stat_folder']):
        os.makedirs(config['stat_folder'])
    with open(save_file_name, 'wb') as f:
        pickle.dump(stat_embryo, f)
        stat_embryo.to_csv(save_file_name_csv)


def cell_graph_network(config):
    '''
    Used to construct the contact relationship at one specific time point. The vertex represents the cell, and there is
    a edge whenever two SegCell contact with each other.
    :param config: parameter configs
    :return :
    '''
    time_point = config['time_point']
    seg_file = os.path.join(config['seg_folder'], config['embryo_name'], config['embryo_name']+"_"+str(time_point).zfill(3)+'_segCell.nii.gz')
    nucleus_loc_file = os.path.join(config["project_folder"], 'TemCellGraph', config['embryo_name'], config['embryo_name']+"_"+str(time_point).zfill(3)+'_nucLoc'+'.txt')  # read nucleus location Data


    #  Load the dictionary of cell and it's coresponding number in the dictionary
    pd_number = pd.read_csv(config["number_dictionary"], names=["name", "label"])
    number_dict = pd.Series(pd_number.label.values, index=pd_number.name).to_dict()
    name_dict = {value: key for key, value in number_dict.items()}

    ## unify the labels in the segmentation and that in the aceTree information
    division_seg, nuc_position = unify_label_seg_and_nuclues(file_lock, time_point, seg_file, config)
    division_seg_save_file = os.path.join(os.path.dirname(seg_file)+'LabelUnified', config['embryo_name'] + "_" + str(time_point).zfill(3)+'_segCell.nii.gz')
    save_nii(division_seg, division_seg_save_file)

    ##  cinstruct graph
    #  add vertex
    with open(nucleus_loc_file, 'rb') as f:
        nucleus_loc = pickle.load(f)
    all_labels = list(np.unique(division_seg))
    all_labels.remove(0)
    point_graph = nx.Graph()
    point_graph.clear()
    for label in all_labels:
        cell_name = name_dict[label]
        point_graph.add_node(cell_name, pos=nucleus_loc[nucleus_loc.nucleus_name==cell_name].iloc[:, 2:5].values[0].tolist())
    #  add connections between SegCell (edge and edge weight)

    relation_graph = add_relation(point_graph, division_seg, name_dict)

    # nx.draw(relation_graph, pos=nuc_position, with_labels=True, node_size=100, font_color='b', edge_cmap=plt.cm.Blues)  # TODO: better visualization on graph
    file_name = os.path.join(config["project_folder"], 'TemCellGraph', config['embryo_name'], config['embryo_name'] + '_T' + str(config['time_point']) + '.txt')
    with open(file_name, 'wb') as f:
        pickle.dump(relation_graph, f)

def unify_label_seg_and_nuclues(file_lock, time_point, seg_file, config):
    '''
    Use acetree nucleus information to unify the segmentation labels in the membrane segmentations.
    :param file_lock: file locker to control parallel computing
    :param time_point: time point of the volume
    :param seg_file: cell segmentation file
    :param config: parameter configs
    :return unify_seg: cell segmentation with labels unified
    :return nuc_positions: nucleus positions with labels
    '''

    pd_number = pd.read_csv(config["number_dictionary"], names=["name", "label"])
    number_dict = pd.Series(pd_number.label.values, index=pd_number.name).to_dict()

    name_dict = {value: key for key, value in number_dict.items()}
    cell_tree = config["cell_tree"]

    df = pd.read_csv(config['acetree_file'])
    df_t = df[df.time==time_point]
    nucleus_names = list(df_t.cell)  # all names based on nucleus location
    nucleus_number = [number_dict[cell_name] for cell_name in nucleus_names]

    ## extract nucleus location information in the aceTree
    nucleus_location = df_t.loc[:, ['z', 'x', 'y']].copy()
    ace_shape = config['image_size'].copy()
    # nucleus_location['z'] = nucleus_location['z'] * config['z_resolution'] / config['xy_resolution']
    # ace_shape[0] = ace_shape[0] * config['z_resolution'] / config['xy_resolution']
    nucleus_location = nucleus_location.values

    ## load seg volume
    seg = nib.load(seg_file).get_data().transpose([2, 1, 0])
    config["res"] = ace_shape[-1] / seg.shape[-1] * config["xy_resolution"]
    nucleus_location_zoom = (nucleus_location * np.array(seg.shape) / np.array(ace_shape)).astype(np.uint16)
    nucleus_location_zoom[:, 0] = seg.shape[0] - nucleus_location_zoom[:, 0]
    # nucleus_location_zoom[:, 0] = seg.shape[0] - nucleus_location_zoom[:, 0]  # the embryo is reversed at z axis
    ####################To save nucleus location Data##########################
    nucleus_loc_to_save = pd.DataFrame.from_dict({'nucleus_label':nucleus_number, 'nucleus_name':nucleus_names,
                                                  f'x_{seg.shape[2]}':nucleus_location_zoom[:, 2], f'y_{seg.shape[1]}':nucleus_location_zoom[:, 1],
                                                  f'z_{seg.shape[0]}':nucleus_location_zoom[:, 0]})
    save_name = os.path.join(config['save_nucleus_folder'], config['embryo_name'], config['embryo_name']+"_"+str(time_point).zfill(3)+'_nucLoc'+'.csv')
    save_name_fast_read = os.path.join(config["project_folder"], 'TemCellGraph', config['embryo_name'], config['embryo_name']+"_"+str(time_point).zfill(3)+'_nucLoc'+'.txt')
    ##################################################
    #  unify the segmentation label
    unify_seg = np.zeros_like(seg)  #TODO: update cell mother daughter's label
    changed_flag = np.zeros_like(seg)  # to label whether a cell has been updated with labels in the nucleus stack.
    nucleus_loc_to_save["volume"] = "" ################### Used for wrting cell information
    nucleus_loc_to_save["surface"] = "" ################### Used for wrting cell information
    nucleus_loc_to_save["note"] = "" ################### Used for wrting cell information
    for i, nucleus_loc in enumerate(list(nucleus_location_zoom)):
        target_label = nucleus_number[i]
        if "Nuc" in nucleus_names[i]:  # ignore all SegCell starting with "Nuc****"
            nucleus_loc_to_save.loc[nucleus_loc_to_save.nucleus_label == target_label, "note"] = "lost_hole"
            update_time_tree(config['embryo_name'], name_dict[target_label], time_point, file_lock, config, add=False)
            continue
        raw_label = seg[nucleus_loc[0], nucleus_loc[1], nucleus_loc[2]]
        update_flag = changed_flag[nucleus_loc[0], nucleus_loc[1], nucleus_loc[2]]
        if raw_label != 0:
            if not update_flag:
                unify_seg[seg == raw_label] = target_label
                changed_flag[seg==raw_label] = 1
                # add volume and surface information
                surface_area = get_surface_area(seg == raw_label)
                nucleus_loc_to_save.loc[nucleus_loc_to_save.nucleus_label == target_label, "volume"] = (seg == raw_label).sum() * (config["res"] ** 3)
                nucleus_loc_to_save.loc[nucleus_loc_to_save.nucleus_label == target_label, "surface"] = surface_area * (config["res"] ** 2)
            else:
                # check whether two labels from the same mother
                another_label = unify_seg[nucleus_loc[0], nucleus_loc[1], nucleus_loc[2]]
                another_mother_name = cell_tree.parent(name_dict[another_label]).tag
                mother_name = cell_tree.parent(name_dict[target_label]).tag
                if another_mother_name == mother_name:
                    mother_label = number_dict[mother_name]
                    ################### add a virtual meother nucleus to the nucleus loc file
                    mother_loc, ch1, ch2 = get_mother_loc(cell_tree, mother_name, nucleus_loc_to_save)
                    if mother_loc is not None:
                        unify_seg[seg == raw_label] = mother_label
                        nucleus_loc_to_save= nucleus_loc_to_save.append({
                            "nucleus_label": mother_label,
                            "nucleus_name": mother_name,
                            f'x_{seg.shape[2]}': mother_loc[0],
                            f'y_{seg.shape[1]}': mother_loc[1],
                            f'z_{seg.shape[0]}': mother_loc[2],
                            "note": "mother"
                        }, ignore_index=True)
                        surface_area = get_surface_area(seg == raw_label)
                        nucleus_loc_to_save.loc[nucleus_loc_to_save.nucleus_label == mother_label, "volume"] = (seg == raw_label).sum() * (config["res"] ** 3)
                        nucleus_loc_to_save.loc[nucleus_loc_to_save.nucleus_label == mother_label, "surface"] = surface_area * (config["res"] ** 2)
                        # update daughter SegCell information
                        nucleus_loc_to_save = update_daughter_info(nucleus_loc_to_save, ch1, ch2, mother_name)
                        update_time_tree(config['embryo_name'], mother_name, time_point, file_lock, config, add=True)
                        update_time_tree(config['embryo_name'], ch1, time_point, file_lock, config, add=False)
                        update_time_tree(config['embryo_name'], ch2, time_point, file_lock, config, add=False)
                    else:
                        # The nucleus is also occupied
                        nucleus_loc_to_save.loc[nucleus_loc_to_save.nucleus_label == target_label, "note"] = "lost_inner1"
                        update_time_tree(config['embryo_name'], name_dict[target_label], time_point, file_lock, config, add=False)
                else:
                    # nucleus is occupied by strangers
                    nucleus_loc_to_save.loc[nucleus_loc_to_save.nucleus_label == target_label, "note"] = "lost_inner2"
                    update_time_tree(config['embryo_name'], name_dict[target_label], time_point, file_lock, config, add=False)
        else:
            # nucleus locates in the background
            nucleus_loc_to_save.loc[nucleus_loc_to_save.nucleus_label==target_label, "note"] = "lost_hole"
            update_time_tree(config['embryo_name'], name_dict[target_label], time_point, file_lock, config, add=False)

    check_folder_exist(save_name)
    nucleus_loc_to_save.to_csv(save_name, index=False) ######################
    check_folder_exist(save_name_fast_read)
    with open(save_name_fast_read, 'wb') as f:
        pickle.dump(nucleus_loc_to_save, f)

    ##  deal with dividing SegCell
    raw_labels = list(seg[nucleus_location_zoom[:, 0], nucleus_location_zoom[:, 1], nucleus_location_zoom[:, 2]])
    repeat_labels = [[i, label] for i, label in enumerate(raw_labels) if raw_labels.count(label) > 1]
    repeat_labels = [x for x in repeat_labels if x[1] != 0]  # Label with 0 is the missed cell #TODO
    # reset all labels to their parent label
    division_seg = unify_seg.copy()
    cell_locations = [list(x) for x in list(nucleus_location_zoom)]
    cell_names = nucleus_names.copy()
    for repeat_label in repeat_labels:
        cell_name = name_dict[nucleus_number[repeat_label[0]]]
        cell_names.remove(cell_name)
        cell_locations.remove(list(nucleus_location_zoom[repeat_label[0]]))
        try:
            parent_node = cell_tree.parent(cell_name)
            parent_label = number_dict[parent_node.tag]
        except:
            pass
        division_seg[unify_seg == number_dict[cell_name]] = parent_label
        if name_dict[parent_label] not in cell_names:
            cell_names.append(name_dict[parent_label])
            cell_locations.append(list(nucleus_location_zoom[repeat_label[0]]))  # TODO: change this when plot in 3D
    nuc_positions = dict(zip(cell_names, cell_locations)) # ABalappap

    return unify_seg, nuc_positions


def add_relation(point_graph, division_seg, name_dict):
    '''
    Add relationship information between SegCell. (contact surface area)
    :param point_graph: point graph of SegCell
    :param division_seg: cell segmentations
    :return point_graph: contact graph between cells
    '''
    if np.unique(division_seg).shape[0] > 2:  # in case there are multiple cells
        contact_pairs, contact_area = get_contact_area(division_seg)
        for i, one_pair in enumerate(contact_pairs):
            point_graph.add_edge(name_dict[one_pair[0]], name_dict[one_pair[1]], area=contact_area[i])

    return point_graph


def get_contact_area(volume):
    '''
    Get the contact volume surface of the segmentation. The segmentation results should be watershed segmentation results
    with a ***watershed line***.
    :param volume: segmentation result
    :return boundary_elements_uni: pairs of SegCell which contacts with each other
    :return contact_area: the contact surface area corresponding to that in the the boundary_elements.
    '''

    cell_mask = volume != 0
    boundary_mask = (cell_mask == 0) & ndimage.binary_dilation(cell_mask)
    [x_bound, y_bound, z_bound] = np.nonzero(boundary_mask)
    boundary_elements = []
    for (x, y, z) in zip(x_bound, y_bound, z_bound):
        neighbors = volume[np.ix_(range(x-1, x+2), range(y-1, y+2), range(z-1, z+2))]
        neighbor_labels = list(np.unique(neighbors))
        neighbor_labels.remove(0)
        if len(neighbor_labels) == 2:
            boundary_elements.append(neighbor_labels)
    boundary_elements_uni = list(np.unique(np.array(boundary_elements), axis=0))
    contact_area = []
    boundary_elements_uni_new = []
    for (label1, label2) in boundary_elements_uni:
        contact_mask = np.logical_and(ndimage.binary_dilation(volume == label1), ndimage.binary_dilation(volume == label2))
        contact_mask = np.logical_and(contact_mask, boundary_mask)
        if contact_mask.sum() > 4:
            verts, faces, _, _ = marching_cubes_lewiner(contact_mask)
            area = mesh_surface_area(verts, faces) / 2
            contact_area.append(area)
            boundary_elements_uni_new.append((label1, label2))
    return boundary_elements_uni_new, contact_area


def construct_stat_embryo(cell_tree, max_time):
    '''
    Construct embryonic statistical DataFrom
    :param cell_tree: cell lineage tree used in the analysis
    :param max_time: the maximum time point we analyze.
    :return:
    '''
    global stat_embryo

    all_names = [cname for cname in cell_tree.expand_tree(mode=Tree.WIDTH)]
    # Get tuble lists with elements from the list
    name_combination = []
    first_level_names = []
    for i, name1 in enumerate(all_names):
        for name2 in all_names[i+1:]:
            if not (cell_tree.is_ancestor(name1, name2) or cell_tree.is_ancestor(name2, name1)):
                first_level_names.append(name1)
                name_combination.append((name1, name2))

    multi_index = pd.MultiIndex.from_tuples(name_combination, names=['cell1', 'cell2'])
    stat_embryo = pd.DataFrame(np.full(shape=(max_time, len(name_combination)), fill_value=np.nan, dtype=np.float32),
                               index=range(1, max_time+1), columns=multi_index)
    # set zero element to express the exist of the specific nucleus
    for cell_name in all_names:
        if cell_name not in first_level_names:
            continue
        try:
            cell_time = cell_tree.get_node(cell_name).data.get_time()
            cell_time = [x for x in cell_time if x <= max_time]
            stat_embryo.loc[cell_time, (cell_name, slice(None))] = 0
        except:
            cell_name


def assemble_result(point_embryo, time_point, number_dict):
    '''
    Assemble results of the embryo at different time points into a single DataFrame in Pandas
    :param point_embryo: embryo information at one time point
    :param time_point: time point of the embryo
    :return st_embryo: statistical shape information. Checked through cell_name1, cell_name2, time.
    '''
    global stat_embryo
    edges_view = point_embryo.edges(data=True)

    for one_edge in edges_view:
        edge_weight = one_edge[2]['area']
        if (one_edge[0], one_edge[1]) in stat_embryo.columns:
            stat_embryo.loc[time_point, (one_edge[0], one_edge[1])] = edge_weight
        elif (one_edge[1], one_edge[0]) in stat_embryo.columns:
            stat_embryo.loc[time_point, (one_edge[1], one_edge[0])] = edge_weight
        else:
            pass

# #   neighbors to text for GUI
#     neighbors_file = os.path.join(config["para"]['seg_folder'], config["para"]['embryo_name']+ "_guiNieghbor", config["para"]['embryo_name'] + "_" +str(time_point).zfill(3)+"_guiNeighbor.txt")
#     if not os.path.isdir(os.path.dirname(neighbors_file)):
#         os.makedirs(os.path.dirname(neighbors_file))
#     for i, cell_name in enumerate(point_embryo.nodes()):
#         neighbor_cells = list(point_embryo.neighbors(cell_name))
#         neighbor_labels = [str(number_dict[name]) for name in neighbor_cells]
#         cell_label = str(number_dict[cell_name])
#         label_str = ",".join(([cell_label] + neighbor_labels))  # first one master cell
#         if i == 0:
#             with open(neighbors_file, "w") as f:
#                 f.write(label_str+"\n")
#         else:
#             with open(neighbors_file, "a") as f:
#                 f.write(label_str+"\n")

    return stat_embryo


def get_mother_loc(cell_tree, mother_name, loc):
    '''
    Get mother nucleus location based on children's location
    :param cell_tree: cell nucleus lineage
    :param mother_name: mother cell name
    :param loc: daughter's location
    :return mother_loc: mother's location
    :return children_name1: first child's name
    :return children_name2: second child's name
    '''
    try:
        children1, children2 = cell_tree.children(mother_name)
        children_name1, children_name2 = [children1.tag, children2.tag]

        mother_loc = (loc[loc.nucleus_name == children_name1].iloc[:, 2:5].values[0] + \
                     loc[loc.nucleus_name == children_name2].iloc[:, 2:5].values[0]) / 2
    except:
        print("test here")
        return None, None, None

    return mother_loc.astype(np.int16).tolist(), children_name1, children_name2


def get_surface_area(cell_mask):
    '''
    get cell surface area
    :param cell_mask: single cell mask
    :return surface_are: cell's surface are
    '''
    # ball_structure = morphology.cube(3)
    # erased_mask = ndimage.binary_erosion(cell_mask, ball_structure, iterations=1)
    # surface_area = np.logical_and(~erased_mask, cell_mask).sum()
    verts, faces, _, _ = marching_cubes_lewiner(cell_mask)
    surface = mesh_surface_area(verts, faces)

    return surface


def update_time_tree(embryo_name, cell_name, time_point, file_lock, config, add=False):
    '''
    Update cell lineage tree. Such as two nuclei in dividing cell are merged into one.
    :param embryo_name: embryo's name
    :param cell_name: cell's name
    :param time_point: time point of the embryo
    :param file_lock: file locker for parallel computing
    :param add: add or remove one cell in the lineage
    :return:
    '''
    file_lock.acquire()
    try:
        with open(os.path.join(config["stat_folder"], "{}_time_tree.txt".format(embryo_name)), 'rb') as f:
            time_tree = pickle.load(f)
        origin_times = time_tree.get_node(cell_name).data.get_time()
        if add:
            origin_times = origin_times + [time_point]
        else:
            origin_times.remove(time_point)
        time_tree.get_node(cell_name).data.set_time(origin_times)
        with open(os.path.join(config["stat_folder"], "{}_time_tree.txt".format(embryo_name)), 'wb') as f:
            pickle.dump(time_tree, f)
    except:
        pass
    finally:
        file_lock.release()


def update_daughter_info(nucleus_loc_info, ch1, ch2, mother):
    '''
    add notes to the nucleus loc file.
    :param nucleus_loc_info: nucleus location including notes
    :param ch1: first child's name
    :param ch2: second child's name
    :param mother: mother's name
    :return nucleus_loc_info: updated nucleus info
    '''
    nucleus_loc_info.loc[nucleus_loc_info.nucleus_name == ch1, ["note", "volume", "surface"]] = ["child_of_{}".format(mother), '', '']
    nucleus_loc_info.loc[nucleus_loc_info.nucleus_name == ch2, ["note", "volume", "surface"]] = ["child_of_{}".format(mother), '', '']

    return nucleus_loc_info