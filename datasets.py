import os
import imghdr
import pathlib
import numpy as np
import torch

import torch.utils.data as data
import torch.nn.functional as F

from utils import Translater
import pandas as pd

from PIL import Image

class ImageDataset(data.Dataset):
    def __init__(self, data_dir, class_num,trnsfm=None):
        super(ImageDataset, self).__init__()

        
        self.data_dir = os.path.expanduser(data_dir)

        #=========================================================================
        # image proceccing information by tensor
        #=========================================================================

        self.transform = trnsfm

        #=========================================================================
        # setting dirpath to loading the images and condition
        #=========================================================================
       
        self.img_paths = self.__load_imgpaths_from_dir(os.path.join(self.data_dir,"images"))
        self.vectors_dir = os.path.join(self.data_dir,"vectors")

        #=========================================================================
        # loading csv file
        #=========================================================================
        
        csv_file_dir = str(pathlib.Path(self.data_dir).parents[0])
        csv_file_path = os.path.join(csv_file_dir,"information.csv")
        self.char_df = pd.read_csv(csv_file_path) 

        #=========================================================================
        # setting class num
        #=========================================================================

        # using these character -> a-z,A-Z
        # so the number of class is 26 + 26 = 52
        self.translater = Translater(num_class=52)

        self.CLASS_NUM = class_num

    def __len__(self):
        return len(self.img_paths)


    def __getitem__(self, index, color_format='RGB'):
        
        # used for loading target vector 
        vector_file_name = os.path.basename(self.img_paths[index]).split(".")[0] + ".npy"

        ###############################################################################
        # Open same name images (real, mask, and inpaint)
        # (To fit the number of images. Some preprocess make gap of number of images )  
        ###############################################################################

        image = Image.open(self.img_paths[index])

        ##############################################################
        
        if self.transform is not None:
            image = self.transform(image)

        image_name_without_ext = os.path.basename(self.img_paths[index]).split(".")[0]
        label = self.char_df.query('image_id == @image_name_without_ext').iat[0,1]
        onehot = self.translater.chr2num(label)

        ##############################################################

        vector_path = os.path.join(self.vectors_dir,vector_file_name)
        vector = np.load(vector_path)
        vector = torch.tensor(vector)

        return  image, onehot, vector


    def __is_imgfile(self, filepath):
        filepath = os.path.expanduser(filepath)
        if os.path.isfile(filepath) and imghdr.what(filepath):
            return True
        return False


    def __load_imgpaths_from_dir(self, dirpath, walk=False):
        imgpaths = []
        dirpath = os.path.expanduser(dirpath)
        if walk:
            for (root, _, files) in os.walk(dirpath):
                for file in files:
                    file = os.path.join(root, file)
                    if self.__is_imgfile(file):
                        imgpaths.append(file)
        else:
            for path in os.listdir(dirpath):
                path = os.path.join(dirpath, path)
                if self.__is_imgfile(path) == False:
                    continue
                imgpaths.append(path)
        return imgpaths