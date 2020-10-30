import json
import os
import sys
import csv
import argparse
from torch.utils.data import DataLoader
from torch.optim import Adadelta, Adam,SGD
from torch.nn import BCELoss, DataParallel,BCEWithLogitsLoss
from torchvision.utils import save_image,make_grid
from PIL import Image
import torchvision.transforms as transforms
import torch
from collections import OrderedDict
import numpy as np
from tqdm import tqdm
from model import VGG
from datasets import ImageDataset
from losses import estimate_network_loss
from utils import (
    sample_random_batch,
    label2img,
    Translater
)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

parser = argparse.ArgumentParser()
parser.add_argument('data_dir')
parser.add_argument('output_dir')

parser.add_argument('--model', type=str, default=None)

parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--saveperiod', type=int, default=5)
parser.add_argument('--input_size', type=int, default=(228,228))
parser.add_argument('--optimizer', type=str, choices=['adadelta', 'adam', 'SGD'], default='adam')
parser.add_argument('--bsize', type=int, default=512)

def main(args):

    # ================================================
    # Preparation
    # ================================================

    args.data_dir = os.path.expanduser(args.data_dir)
    args.result_dir = os.path.expanduser(args.output_dir)

    # create result directory (if necessary)
    if os.path.exists(args.result_dir) == False:
        os.makedirs(args.result_dir)

    output_image_dir = os.path.join(args.result_dir,'images')

    if os.path.exists(output_image_dir) == False:
        os.makedirs(output_image_dir)

    output_vector_dir = os.path.join(args.result_dir,'vectors')

    if os.path.exists(output_vector_dir) == False:
        os.makedirs(output_vector_dir)
    
    # dataset 
    trnsfm = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize(args.input_size),
        transforms.ToTensor()
    ])

    dset = ImageDataset(os.path.join(args.data_dir, 'data'), trnsfm =trnsfm,class_num = 52)
    data_loader = DataLoader(dset, batch_size=1, shuffle=True)

    n_dimention = 100
    n_class = 52

    translater = Translater(num_class=52)

    model = VGG(
        input_img_shape=(1, args.input_size[0],args.input_size[1]),
        n_class =n_class,
        n_dimention=n_dimention)
    model.load_state_dict(torch.load(args.model,map_location='cpu'))
    model.eval()

    csv_filename = os.path.join(args.result_dir,"eval.csv")
    Header = ["image_id","char_label"]


    with open(csv_filename,'w') as f:
        writer = csv.DictWriter(f,fieldnames=Header)
        writer.writeheader()
        # eval and save vectorfile
        torch.set_grad_enabled(False)
        for batch_index, (Images, Labels, Vectors) in enumerate(data_loader):
            # forward
            Images = Images
            condition = label2img(
                shape=(Images.shape[0],Labels.shape[2],Images.shape[2],Images.shape[3]),
                labels = Labels
            )

            input = torch.cat((Images, condition), dim=1)
            output = model(input)

            output = output.numpy()
            #Vectors = Vectors.numpy()
            imgpath = os.path.join(output_image_dir,'{}.png' .format(batch_index))
            save_image(Images,imgpath)
            np.save(os.path.join(output_vector_dir, '{}'.format(batch_index)),output[0])

            labels_onehot = Labels.numpy()

            for char_id,num in enumerate(labels_onehot[0][0]):
                if num == 1:
                    label_id = char_id

            writer.writerow({"image_id":batch_index,"char_label":translater.num2chr(label_id)})

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
