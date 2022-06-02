import os
import glob

import numpy as np
import pandas as pd
from skimage import io

import torch
import torchvision
from torch.utils.data import Dataset

import zipfile
from util import utils_np

'''
'''

class ImageStackDataset(Dataset):
    def __init__(self, csv_path:str, root_dir:str, transform:torchvision.transforms=None, 
                 T_channel:bool=False, dynamic_env:bool=False, pred_traj:bool=False):
        '''
        Args:
            csv_path: Path to the CSV file with dataset info.
            root_dir: Directory with all image folders.
                      root_dir - video_folder - imgs
        '''
        super().__init__()
        self.info_frame = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.tr = transform

        self.with_T = T_channel
        self.dyn_env = dynamic_env
        self.pred_traj = pred_traj

        self.ext = '.jpg'  # should '.'
        self.csv_str = 'p' # in csv files, 'p' means position

        self.input_len = len([x for x in list(self.info_frame) if self.csv_str in x]) # length of input time step
        self.img_shape = self.check_img_shape()

    def __len__(self):
        return len(self.info_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input_img = np.empty(shape=[self.img_shape[0],self.img_shape[1],0])
        info = self.info_frame.iloc[idx]
        index = info['index']
        traj = []
        if self.dyn_env:
            for i in range(self.input_len):
                img_name = info['p{}'.format(i)].split('_')[-1] + self.ext
                img_path = os.path.join(self.root_dir, str(index), img_name)
                this_x = float(info['p{}'.format(i)].split('_')[0])
                this_y = float(info['p{}'.format(i)].split('_')[1])
                traj.append([this_x,this_y])

                image = self.togray(io.imread(img_path))
                obj_map = utils_np.np_gaudist_map((this_x, this_y), np.zeros_like(image), sigmas=[20,20])
                input_img = np.concatenate((input_img, obj_map[:,:,np.newaxis], image[:,:,np.newaxis]), axis=2)
        else:
            img_name = f'{info["index"]}.{self.ext}'
            img_path = os.path.join(self.root_dir, str(index), img_name)
            image = self.togray(io.imread(img_path))
            for i in range(self.input_len):
                position = info['p{}'.format(i)]
                this_x = float(position.split('_')[0])
                this_y = float(position.split('_')[1])
                traj.append([this_x,this_y])

                obj_map = utils_np.np_gaudist_map((this_x, this_y), np.zeros_like(image), sigmas=[20,20])
                input_img = np.concatenate((input_img, obj_map[:,:,np.newaxis]), axis=2)
            input_img = np.concatenate((input_img, image[:,:,np.newaxis]), axis=2)
        
        if self.with_T:
            T_channel = np.ones(shape=[self.img_shape[0],self.img_shape[1],1])*self.T # T_channel
            input_img = np.concatenate((input_img, T_channel), axis=2)                # T_channel

        if self.pred_traj:
            label_name_list = [x for x in list(self.info_frame) if 'T' in x]
            label_list = list(info[label_name_list].values)
            label = [(float(x.split('_')[0]), float(x.split('_')[1])) for x in label_list]
        else:
            label = [(float(info['T'].split('_')[0]), float(info['T'].split('_')[1]))]

        sample = {'image':input_img, 'label':label}
        if self.tr:
            sample = self.tr(sample)
        sample['index'] = index
        sample['traj'] = traj
        sample['time'] = info['t']

        return sample

    def rescale_label(self, label, original_scale): # x,y & HxW
        current_scale = self.check_img_shape()
        rescale = (current_scale[0]/original_scale[0] , current_scale[1]/original_scale[1])
        return (label[0]*rescale[1], label[1]*rescale[0])

    def togray(self, image):
        if (len(image.shape)==2):
            return image
        elif (len(image.shape)==3) and (image.shape[2]==1):
            return image[:,:,0]
        else:
            image = image[:,:,:3] # ignore alpha
            img = image[:,:,0]/3 + image[:,:,1]/3 + image[:,:,2]/3
            return img

    def check_img_shape(self):
        info = self.info_frame.iloc[0]
        img_name = str(info['t']) + self.ext
        video_folder = info['index']
        img_path = os.path.join(self.root_dir, video_folder, img_name)
        image = self.togray(io.imread(img_path))
        return image.shape


class ImageStackDatasetZIP(Dataset):
    def __init__(self, zip_path:str, csv_path:str, root_dir:str, transform:torchvision.transforms=None, 
                 T_channel:bool=False, dynamic_env:bool=False, pred_traj:bool=False):
        '''
        Args:
            zip_path: Path to the zip file with all files
            csv_path: Path to the CSV file with dataset info.
            root_dir: Directory with all image folders.
                      root_dir - video_folder - imgs
        '''
        super().__init__()
        self.archive = zipfile.ZipFile(zip_path, 'r')

        self.info_frame = pd.read_csv(self.archive.open(csv_path))
        self.root_dir = root_dir
        self.tr = transform

        self.with_T = T_channel
        self.dyn_env = dynamic_env
        self.pred_traj = pred_traj

        self.ext = '.jpg'
        self.csv_str = 'p' # in csv files, 'p' means position

        self.input_len = len([x for x in list(self.info_frame) if self.csv_str in x]) # length of input time step
        self.img_shape = self.check_img_shape()

    def __len__(self):
        return len(self.info_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input_img = np.empty(shape=[self.img_shape[0],self.img_shape[1],0])
        info = self.info_frame.iloc[idx]
        index = info['index']
        traj = []
        if self.dyn_env:
            for i in range(self.input_len):
                img_name = info['p{}'.format(i)].split('_')[-1] + self.ext
                img_path = os.path.join(self.root_dir, str(index), img_name)
                this_x = float(info['p{}'.format(i)].split('_')[0])
                this_y = float(info['p{}'.format(i)].split('_')[1])
                traj.append([this_x,this_y])

                image = self.togray(io.imread(self.archive.open(img_path)))
                obj_map = utils_np.np_gaudist_map((this_x, this_y), np.zeros_like(image), sigmas=[20,20])
                input_img = np.concatenate((input_img, obj_map[:,:,np.newaxis], image[:,:,np.newaxis]), axis=2)
        else:
            img_name = f'{info["index"]}.{self.ext}'
            img_path = os.path.join(self.root_dir, str(index), img_name)
            image = self.togray(io.imread(self.archive.open(img_path)))
            for i in range(self.input_len):
                position = info['p{}'.format(i)]
                this_x = float(position.split('_')[0])
                this_y = float(position.split('_')[1])
                traj.append([this_x,this_y])

                obj_map = utils_np.np_gaudist_map((this_x, this_y), np.zeros_like(image), sigmas=[20,20])
                input_img = np.concatenate((input_img, obj_map[:,:,np.newaxis]), axis=2)
            input_img = np.concatenate((input_img, image[:,:,np.newaxis]), axis=2)
        
        if self.with_T:
            T_channel = np.ones(shape=[self.img_shape[0],self.img_shape[1],1])*self.T # T_channel
            input_img = np.concatenate((input_img, T_channel), axis=2)                # T_channel

        if self.pred_traj:
            label_name_list = [x for x in list(self.info_frame) if 'T' in x]
            label_list = list(info[label_name_list].values)
            label = [(float(x.split('_')[0]), float(x.split('_')[1])) for x in label_list]
        else:
            label = [(float(info['T'].split('_')[0]), float(info['T'].split('_')[1]))]

        sample = {'image':input_img, 'label':label}
        if self.tr:
            sample = self.tr(sample)
        sample['index'] = index
        sample['traj'] = traj
        sample['time'] = info['t']

        return sample

    def rescale_label(self, label, original_scale): # x,y & HxW
        current_scale = self.check_img_shape()
        rescale = (current_scale[0]/original_scale[0] , current_scale[1]/original_scale[1])
        return (label[0]*rescale[1], label[1]*rescale[0])

    def togray(self, image):
        if (len(image.shape)==2):
            return image
        elif (len(image.shape)==3) and (image.shape[2]==1):
            return image[:,:,0]
        else:
            image = image[:,:,:3] # ignore alpha
            img = image[:,:,0]/3 + image[:,:,1]/3 + image[:,:,2]/3
            return img

    def check_img_shape(self):
        info = self.info_frame.iloc[0]
        img_name = str(info['t']) + self.ext
        video_folder = info['index']
        img_path = os.path.join(self.root_dir, video_folder, img_name)
        image = self.togray(io.imread(img_path))
        return image.shape


# XXX deprecate
class ImageStackDatasetSDDtr(Dataset): # for trajectory
    def __init__(self, csv_path, root_dir, ext='.jpg', transform=None, T_channel=None):
        '''
        Args:
            csv_path: Path to the CSV file with dataset info.
            root_dir: Directory with all image folders.
                      root_dir - video_folder - imgs
        '''
        super().__init__()
        self.info_frame = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.tr = transform
        self.ext = ext

        self.nc = len([x for x in list(self.info_frame) if 't' in x]) # number of image channels in half
        self.img_shape = self.check_img_shape()

    def __len__(self):
        return len(self.info_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input_img = np.empty(shape=[self.img_shape[0],self.img_shape[1],0])
        info = self.info_frame.iloc[idx]
        index = info['index']
        traj = []
        for i in range(self.nc):
            img_name = info[f't{i}'].split('_')[0] + self.ext
            video_idx = info['index']

            img_path = os.path.join(self.root_dir, video_idx, img_name)

            csv_name = glob.glob(os.path.join(self.root_dir, video_idx, '*.csv'))
            original_scale = os.path.basename(csv_name[0]).split('.')[0]
            original_scale = (int(original_scale.split('_')[0]), int(original_scale.split('_')[1])) # HxW

            time_step = int(info[f't{i}'].split('_')[0])
            this_x = float(info[f't{i}'].split('_')[1])
            this_y = float(info[f't{i}'].split('_')[2])
            traj.append([this_x,this_y])

            image = self.togray(io.imread(img_path))
            input_img = np.concatenate((input_img, image[:,:,np.newaxis]), axis=2)

            white_canvas = np.zeros_like(image)
            # obj_coords = self.rescale_label((this_x, this_y), original_scale)
            obj_coords = (this_x, this_y)
            obj_map = utils_np.np_gaudist_map(obj_coords, white_canvas, sigmas=[20,20])
            input_img = np.concatenate((input_img, obj_map[:,:,np.newaxis]), axis=2)

        label_name_list = [x for x in list(self.info_frame) if 'T' in x]
        label_list = list(info[label_name_list].values)
        label_list = [(float(x.split('_')[0]), float(x.split('_')[1])) for x in label_list]
        label = dict(zip(label_name_list, label_list))
        sample = {'image':input_img, 'label':label}

        if self.tr:
            sample = self.tr(sample)

        sample['index'] = index
        sample['traj'] = traj
        sample['time'] = time_step

        return sample

    def rescale_label(self, label, original_scale): # x,y & HxW
        current_scale = self.check_img_shape()
        rescale = (current_scale[0]/original_scale[0] , current_scale[1]/original_scale[1])
        return (label[0]*rescale[1], label[1]*rescale[0])

    def togray(self, image):
        if (len(image.shape)==2):
            return image
        elif (len(image.shape)==3) and (image.shape[2]==1):
            return image[:,:,0]
        else:
            image = image[:,:,:3] # ignore alpha
            img = image[:,:,0]/3 + image[:,:,1]/3 + image[:,:,2]/3
            return img

    def check_img_shape(self):
        info = self.info_frame.iloc[0]
        img_name = info['t0'].split('_')[0] + self.ext
        video_folder = info['index']
        img_path = os.path.join(self.root_dir, video_folder, img_name)
        image = self.togray(io.imread(img_path))
        return image.shape
# XXX deprecate
class ImageStackDatasetSDDtr_ZIP(Dataset): # for trajectory
    def __init__(self, zip_path, csv_path, root_dir, ext='.jpg', transform=None, T_channel=None):
        '''
        Args:
            zip_path: Path (absolute) to the ZIP file with everything
            csv_path: Path (relative) to the CSV file with dataset info.
            root_dir: Directory (relative) with all image folders.
                      root_dir - obj_folder - obj & other
        '''
        super().__init__()
        self.archive = zipfile.ZipFile(zip_path, 'r')

        self.info_frame = pd.read_csv(self.archive.open(csv_path))
        self.root_dir = root_dir
        self.tr = transform
        self.ext = ext

        self.nc = len([x for x in list(self.info_frame) if 't' in x]) # number of image channels in half
        self.img_shape = self.check_img_shape()

    def __len__(self):
        return len(self.info_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input_img = np.empty(shape=[self.img_shape[0],self.img_shape[1],0])
        info = self.info_frame.iloc[idx]
        index = info['index']
        traj = []
        for i in range(self.nc):
            img_name = info[f't{i}'].split('_')[0] + self.ext
            video_idx = info['index']

            img_path = os.path.join(self.root_dir, video_idx, img_name)

            csv_name = [x for x in self.archive.namelist() if ((video_idx in x)&('csv' in x))]
            original_scale = os.path.basename(csv_name[0]).split('.')[0]
            original_scale = (int(original_scale.split('_')[0]), int(original_scale.split('_')[1])) # HxW

            time_step = int(info[f't{i}'].split('_')[0])
            this_x = float(info[f't{i}'].split('_')[1])
            this_y = float(info[f't{i}'].split('_')[2])
            traj.append([this_x,this_y])

            image = self.togray(io.imread(self.archive.open(img_path)))
            input_img = np.concatenate((input_img, image[:,:,np.newaxis]), axis=2)

            white_canvas = np.zeros_like(image)
            # obj_coords = self.rescale_label((this_x, this_y), original_scale)
            obj_coords = (this_x, this_y)
            obj_map = utils_np.np_gaudist_map(obj_coords, white_canvas, sigmas=[20,20])
            input_img = np.concatenate((input_img, obj_map[:,:,np.newaxis]), axis=2)

        label_name_list = [x for x in list(self.info_frame) if 'T' in x]
        label_list = list(info[label_name_list].values)
        label_list = [(float(x.split('_')[0]), float(x.split('_')[1])) for x in label_list]
        label = dict(zip(label_name_list, label_list))
        sample = {'image':input_img, 'label':label}

        if self.tr:
            sample = self.tr(sample)

        sample['index'] = index
        sample['traj'] = traj
        sample['time'] = time_step

        return sample

    def rescale_label(self, label, original_scale): # x,y & HxW
        current_scale = self.check_img_shape()
        rescale = (current_scale[0]/original_scale[0] , current_scale[1]/original_scale[1])
        return (label[0]*rescale[1], label[1]*rescale[0])

    def togray(self, image):
        if (len(image.shape)==2):
            return image
        elif (len(image.shape)==3) and (image.shape[2]==1):
            return image[:,:,0]
        else:
            image = image[:,:,:3] # ignore alpha
            img = image[:,:,0]/3 + image[:,:,1]/3 + image[:,:,2]/3
            return img

    def check_img_shape(self):
        info = self.info_frame.iloc[0]
        img_name = info['t0'].split('_')[0] + self.ext
        video_folder = info['index']
        img_path = os.path.join(self.root_dir, video_folder, img_name)
        image = self.togray(io.imread(self.archive.open(img_path)))
        return image.shape
# XXX deprecate
class ImageStackDatasetSDDtr_SEG(Dataset): # for trajectory
    def __init__(self, csv_path, root_dir, ext=None, transform=None, T_channel=None):
        '''
        Args:
            csv_path: Path to the CSV file with dataset info.
            root_dir: Directory with all image folders.
                      root_dir - video_folder - imgs
        '''
        super().__init__()
        self.info_frame = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.tr = transform

        self.nc = len([x for x in list(self.info_frame) if 't' in x]) # number of image channels in half
        self.img_shape = self.check_img_shape()

    def __len__(self):
        return len(self.info_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input_img = np.empty(shape=[self.img_shape[0],self.img_shape[1],0])
        info = self.info_frame.iloc[idx]
        video_idx = info['index']
        traj = []
        for i in range(self.nc):
            csv_name = glob.glob(os.path.join(self.root_dir, video_idx, '*.csv'))
            original_scale = os.path.basename(csv_name[0]).split('.')[0]
            original_scale = (int(original_scale.split('_')[0]), int(original_scale.split('_')[1])) # HxW

            time_step = int(info[f't{i}'].split('_')[0])
            this_x = float(info[f't{i}'].split('_')[1])
            this_y = float(info[f't{i}'].split('_')[2])
            traj.append([this_x,this_y])

            white_canvas = np.zeros(self.img_shape)
            # obj_coords = self.rescale_label((this_x, this_y), original_scale)
            obj_coords = (this_x, this_y)
            obj_map = utils_np.np_gaudist_map(obj_coords, white_canvas, sigmas=[20,20])
            input_img = np.concatenate((input_img, obj_map[:,:,np.newaxis]), axis=2)

        img_ref = self.togray(io.imread(os.path.join(self.root_dir, video_idx, 'reference.jpg')))
        img_seg = self.togray(io.imread(os.path.join(self.root_dir, video_idx, 'label.png')))
        input_img = np.concatenate((input_img, img_ref[:,:,np.newaxis]), axis=2)
        input_img = np.concatenate((input_img, img_seg[:,:,np.newaxis]), axis=2)

        label_name_list = [x for x in list(self.info_frame) if 'T' in x]
        label_list = list(info[label_name_list].values)
        label_list = [(float(x.split('_')[0]), float(x.split('_')[1])) for x in label_list]
        label = dict(zip(label_name_list, label_list))
        sample = {'image':input_img, 'label':label}

        if self.tr:
            sample = self.tr(sample)

        sample['index'] = video_idx
        sample['traj'] = traj
        sample['time'] = time_step

        return sample

    def rescale_label(self, label, original_scale): # x,y & HxW
        current_scale = self.check_img_shape()
        rescale = (current_scale[0]/original_scale[0] , current_scale[1]/original_scale[1])
        return (label[0]*rescale[1], label[1]*rescale[0])

    def togray(self, image):
        if (len(image.shape)==2):
            return image
        elif (len(image.shape)==3) and (image.shape[2]==1):
            return image[:,:,0]
        else:
            image = image[:,:,:3] # ignore alpha
            img = image[:,:,0]/3 + image[:,:,1]/3 + image[:,:,2]/3
            return img

    def check_img_shape(self, ref_img='reference.jpg'):
        img_path = os.path.join(self.root_dir, ref_img)
        image = self.togray(io.imread(img_path))
        return image.shape
