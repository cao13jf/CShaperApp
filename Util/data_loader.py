from __future__ import absolute_import, print_function

from tqdm import tqdm
from Util.data_process import *


class DataLoader():
    def __init__(self, config):
        """
        Initialize the calss instance
        :param config: a dictionary representing parameters
        """
        self.config = config
        self.data_root = config['data_root'] if type(config['data_root']) is list else [config['data_root']]  # Save as list
        self.with_ground_truth = config.get('with_ground_truth', False)
        self.data_names = config.get('data_names', None)
        self.max_time = config.get("max_time", -1)
        ##  Data augmentation
        self.with_translate = config.get('with_translate', False)
        self.with_scale = config.get('with_scale', False)
        self.with_rotation = config.get('with_rotation', False)
        self.with_flip = config.get('with_flip', False)
        # Get EDT pre-processing information
        self.label_edt_transform = config.get('label_edt_transform', False)
        self.valid_edt_width = config.get('valid_edt_value', 30)
        self.label_edt_discrete = config.get('label_edt_discrete', False)
        self.edt_discrete_num = config.get('edt_discrete_num', 16)

        self.volumes_lists = []
        self.raw_pathes = []
        self.mask_pathes = []
        self.embryo_names = []


    def __get_embryo_names(self):
        """
        get the list of embryo names, if self.data_names id not None, then load embryo
        names from that file, otherwise search all the names automatically in data_root
        """
        # use pre-defined patient names
        embryo_names = os.listdir(self.data_root[0])
        embryo_names = [name for name in embryo_names if 'plc' in name.lower()]
        if (self.data_names is not None):
            embryo_names = self.data_names

        return embryo_names

    def __load_one_volume(self, file_folder, file_name):
        """
        load volume data.
        para volume_dir: file location;
        d
        """
        # for membrane Data
        assert (file_name is not None)
        volume = load_3d_volume_as_array(os.path.join(file_folder, file_name))
        return volume, file_name

    def load_data(self):
        """
        load all the training/testing Data
        """
        self.embryo_names = self.__get_embryo_names()
        assert (len(self.embryo_names) > 0)  # Exit with empty name list

        for one_embryo_name in self.embryo_names:
            raw_path = os.path.join(self.data_root[0], one_embryo_name, 'RawMemb')
            mask_path = os.path.join(self.data_root[0], one_embryo_name, 'SegMemb')
            volumes_lists = os.listdir(raw_path)
            volumes_lists.sort()
            self.max_time = len(volumes_lists) if self.max_time <= 0 else self.max_time

            # ================= record data ======================================
            self.volumes_lists = self.volumes_lists + volumes_lists[:self.max_time]
            self.raw_pathes = self.raw_pathes + [raw_path] * self.max_time
            self.mask_pathes = self.mask_pathes + [mask_path] * self.max_time
            self.embryo_names = self.embryo_names + [one_embryo_name] * self.max_time


    def get_subimage_batch(self):
        """
        sample a batch of image patches for segmentation. Only used for training
        """
        batch = self.__get_one_batch()
        return batch

    def __get_one_batch(self):  # internal method without being overwrited
        """
        get a batch from training Data
        """
        batch_size = self.config['batch_size']
        data_shape = self.config['data_shape']
        label_shape = self.config['label_shape']
        data_slice_number = data_shape[0]  # slice number locates at first
        label_slice_number = label_shape[0]
        slice_direction = self.config.get('slice_direction', 'axial')  # axial, sagittal, coronal or random

        # return batch size: [batch_size, slice_num, slice_h, slice_w, moda_chnl]
        data_batch = []
        label_batch = []
        if (slice_direction == 'random'):  # random slice order
            directions = ['axial', 'sagittal', 'coronal']
            idx = random.randint(0, 2)
            slice_direction = directions[idx]

        batch_data_idx = random.choices(range(len(self.volumes_lists)), k=batch_size)
        for data_idx in batch_data_idx:
            data_volume = self.get_raw_data(data_idx)
            if (self.with_ground_truth):
                label_volume = self.get_label_data(data_idx)
                ## Data augmentation
                [data_volume, label_volume] = self.augment_data(
                    [data_volume, label_volume], issegs=[False, True])

            [data_volume] = self.augment_data([data_volume], issegs=[False])
            transposed_volume = transpose_volumes(data_volume, slice_direction)  # Transpose the direction of volume
            volume_shape = transposed_volume.shape
            sub_data_shape = [data_slice_number, data_shape[1], data_shape[2]]
            sub_label_shape = [label_slice_number, label_shape[1], label_shape[2]]
            center_point = get_random_crop_center(volume_shape, sub_label_shape)
            sub_data = crop_from_volume(transposed_volume, center_point, sub_data_shape)

            if (self.with_ground_truth):
                tranposed_label = transpose_volumes(label_volume, slice_direction)
                sub_label = crop_from_volume(tranposed_label, center_point, sub_label_shape, fill='zero')

            data_batch.append([sub_data])
            if (self.with_ground_truth):
                label_batch.append([sub_label])

        data_batch = np.asarray(data_batch, np.float32)
        label_batch = np.asarray(label_batch, np.float)
        batch = {}

        batch['images'] = np.transpose(data_batch, [0, 2, 3, 4, 1])
        batch['labels'] = np.transpose(label_batch, [0, 2, 3, 4, 1])

        return batch

    def get_total_image_number(self):
        """
        get the toal number of images
        """
        return len(self.volumes_lists)

    def get_image_data_with_name(self, i):
        """
        Used for testing, get one image Data and patient name
        """
        # load RawMemb volume
        volume, volume_name = self.__load_one_volume(self.raw_pathes[i], self.volumes_lists[i])

        bbmin, bbmax = [0, 0, 0], [y - 1 for y in volume.shape]
        bbox = [bbmin, bbmax]  # Box size
        volume = itensity_normalize_one_volume(volume)
        volume_size = volume.shape

        return [volume, self.volumes_lists[i], self.embryo_names[i], bbox, volume_size]

    def get_raw_data(self, idx):
        data, _ = self.__load_one_volume(self.raw_pathes[idx], self.volumes_lists[idx])

        return data

    def get_label_data(self, idx):
        mask_name = "_".join(self.volumes_lists[idx].split(".")[0].split("_")[0:2] + ["segMemb.nii.gz"])
        label, _ = self.__load_one_volume(self.mask_pathes[idx], mask_name)

        # check whether implement EDT on label
        if (self.label_edt_transform):
            if (self.label_edt_discrete):
                label = binary_to_EDT_3D(label, self.valid_edt_width, self.edt_discrete_num).astype(
                    np.uint8)
            else:
                label = binary_to_EDT_3D(label, self.valid_edt_width).astype(np.uint8)

        return label

    '''
    functions for augmentation
    '''
    def augment_data(self, imgs, issegs=[False]):
        random_nums = [random.random() for i in range(4)]  # control  transform ratio
        out_imgs = []
        for i, img in enumerate(imgs):
            if (self.with_translate) and (random_nums[0] > 0.5):
                offset = [random.randint(-10, 10), random.randint(-10, 10), random.randint(-5, 5)]
                img = self.translate_it(img, offset, isseg=issegs[i])

            if (self.with_scale) and (random_nums[1] > 0.5):  # with random probability
                factor = [random.uniform(0.9, 1.1), random.uniform(0.9, 1.1), random.uniform(0.9, 1.1)]
                img = self.scale_it(img, factor, isseg=issegs[i])

            if (self.with_rotation) and (random_nums[2] > 0.5):
                theta = random.uniform(30, -30)
                axes = (1, 0)
                img = self.rotate_it(img, theta, axes, isseg=issegs[i])

            if (self.with_flip) and (random_nums[3] > 0.5):
                axes = random.choice([0, 1, 2])
                img = self.flip_it(img, axes)
            out_imgs.append(img)

        return out_imgs

    def translate_it(self, img, offset=[10, 10, 2], isseg=False, mode='nearest'):
        order = 0 if isseg == True else 5

        return ndimage.interpolation.shift(img, (int(offset[0]), int(offset[1]), int(offset[2])), order=order,
                                           mode=mode)

    def scale_it(self, img, factor=[0.8, 0.8, 0.9], isseg=False, mode='nearest'):
        order = 0 if isseg == True else 3

        height, width, depth = img.shape
        zheight = int(np.round(factor * height))
        zwidth = int(np.round(factor * width))
        zdepth = depth

        if factor < 1.0:
            newimg = np.zeros_like(img)
            row = (height - zheight) // 2
            col = (width - zwidth) // 2
            layer = (depth - zdepth) // 2
            newimg[row:row + zheight, col:col + zwidth, layer:layer + zdepth] = ndimage.interpolation.zoom(
                img,
                (float(factor[0]), float(factor[1]), float(factor[2])),
                order=order,
                mode=mode)[0:zheight, 0:zwidth, 0:zdepth]

            return newimg

        elif factor > 1.0:
            row = (zheight - height) // 2
            col = (zwidth - width) // 2
            layer = (zdepth - depth) // 2

            newimg = ndimage.interpolation.zoom(img[row:row + zheight, col:col + zwidth, layer:layer + zdepth],
                                                (float(factor[0]), float(factor[1]), float(factor[2])), order=order,
                                                mode=mode)
            extrah = (newimg.shape[0] - height) // 2
            extraw = (newimg.shape[1] - width) // 2
            extrad = (newimg.shape[2] - depth) // 2
            newimg = newimg[extrah:extrah + height, extraw:extraw + width, extrad:extrad + depth]

            return newimg

        else:
            return img

    def rotate_it(self, img, theta, axes=(1, 0), isseg=False, mode='constant'):
        order = 0 if isseg == True else 5

        return ndimage.rotate(img, float(theta), axes=axes, reshape=False, order=order, mode=mode)

    def flip_it(self, img, axes=0):

        return np.flip(img, axes)

