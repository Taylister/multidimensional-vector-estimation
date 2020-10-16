import json
import os
import sys
import argparse
from torch.utils.data import DataLoader
from torch.optim import Adadelta, Adam,SGD
from torch.nn import BCELoss, DataParallel,BCEWithLogitsLoss
from torchvision.utils import save_image,make_grid
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np
from tqdm import tqdm
from model import Network
from datasets import ImageDataset
from losses import estimate_network_loss
from utils import (
    sample_random_batch,
    label2img
)

from tensorboardX import SummaryWriter

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

parser = argparse.ArgumentParser()
parser.add_argument('data_dir')
parser.add_argument('result_dir')

parser.add_argument('--model_weight', type=str, default=None)

parser.add_argument('--epoch', type=int, default=1000)
parser.add_argument('--saveperiod', type=int, default=5)
parser.add_argument('--input_size', type=int, default=(60,60))
parser.add_argument('--optimizer', type=str, choices=['adadelta', 'adam', 'SGD'], default='SGD')
parser.add_argument('--bsize', type=int, default=64)

def main(args):

    # ================================================
    # Preparation
    # ================================================

    args.data_dir = os.path.expanduser(args.data_dir)
    args.result_dir = os.path.expanduser(args.result_dir)
    
    # load pretrained model if the model exist
    if args.model_weight != None:
        args.model_weight = os.path.expanduser(args.model_weight)
    if torch.cuda.is_available() == False:
        raise Exception('At least one gpu must be available.')
    else:
        gpu = torch.device('cuda')

    # create result directory (if necessary)
    if os.path.exists(args.result_dir) == False:
        os.makedirs(args.result_dir)

    # dataset 
    trnsfmColor = transforms.Compose([
        transforms.Resize(args.input_size),
        transforms.ToTensor()
    ])

    torch.cuda.empty_cache()

    print("loading dataset... (it may take a few minutes)")
    train_dset = ImageDataset(os.path.join(args.data_dir, 'datasets'), trnsfm =trnsfmColor,class_num = 52)
    #test_dset = ImageDataset(os.path.join(args.data_dir, 'test'), trnsfm1 =trnsfmColor, trnsfm2 = trnsfmGray, class_num = 63)
    
    train_loader = DataLoader(train_dset, batch_size=args.bsize, shuffle=True)


    # ================================================
    # Training Phase 
    # ================================================
    fc_shapes = [256]
    block = [1,2,3,4,5][1]
    n_dimention = 100

    model = Network(block=block,
                    input_shape=(3, args.input_size[0],args.input_size[1]),
                    fc_shapes=fc_shapes,
                    n_dimention=n_dimention)

    if args.model_weight != None:
        model.load_state_dict(torch.load(args.model_weight, map_location='cpu'))
    if args.optimizer == 'adadelta':
        opt = Adadelta(model.parameters())
    elif args.optimizer == 'adam':
        opt = Adam(model.parameters(),lr=0.5)
    else:
        opt = SGD(model.parameters(),lr=0.1)

    model = model.to(gpu)

    writer = SummaryWriter()

    # training
    pbar = tqdm(total=args.epoch)
    while pbar.n < args.epoch:
        model.train()
        # training params and calculate Loss of Network
        train_loss_acm = 0
        for batch_index, (Images, Labels, Vectors) in enumerate(train_loader):
            # forward
            Images = Images.to(gpu)
            #Labels
            """
            condition_g = label2img(
                shape=(X.shape[0],Labels.shape[2],X.shape[2]//2,X.shape[3]//2),
                labels = Labels
            )
            """
            Vectors = Vectors.to(gpu)
            input = Images
            output = model(input)

            loss = estimate_network_loss(output, Vectors)
            train_loss_acm += loss.item() * input.size(0)
            
            # backward
            loss.backward()
            # optimize
            opt.step()

            # clear grads
            opt.zero_grad()

            # tensorboard用log出力設定1[ポイント3]
            writer.add_scalar('data/train_loss', loss.item(), pbar.n*len(train_loader)+batch_index)
        
        train_loss_acm /= len(train_loader.dataset)
        # update progbar
        pbar.set_description('train loss: %.5f' % train_loss_acm)
        pbar.update()
        """
        with torch.no_grad():
            Images, Labels, Vectors= sample_random_batch(test_dset, batch_size=32)
        """
    pbar.close()
    writer.close()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
