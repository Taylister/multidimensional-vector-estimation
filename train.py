import json
import os
import sys
import argparse
from torch.utils.data import DataLoader
from torch.optim import Adadelta, Adam
from torch.nn import BCELoss, DataParallel,BCEWithLogitsLoss
from torchvision.utils import save_image,make_grid
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np
from tqdm import tqdm
from models import CompletionNetwork, ContextDiscriminator
from datasets import ImageDataset
from losses import completion_network_loss
from utils import (
    gen_input_mask,
    gen_hole_area,
    crop,
    sample_random_batch,
    poisson_blend,
    extract_mask_region,
    label2img
)

from tensorboardX import SummaryWriter

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

parser = argparse.ArgumentParser()
parser.add_argument('data_dir')
parser.add_argument('result_dir')

parser.add_argument('--model_weight', type=str, default=None)

parser.add_argument('--epoch', type=int, default=10000)
parser.add_argument('--saveperiod_1', type=int, default=5)
parser.add_argument('--input_size', type=int, default=(60,60))
parser.add_argument('--optimizer', type=str, choices=['adadelta', 'adam'], default='adadelta')
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
        transforms.Resize(args.cn_input_size),
        transforms.ToTensor()
    ])

    trnsfmGray = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize(args.cn_input_size),
        transforms.ToTensor()
    ])

    torch.cuda.empty_cache()

    print("loading dataset... (it may take a few minutes)")
    train_dset = ImageDataset(os.path.join(args.data_dir, 'train'), trnsfm1 =trnsfmColor, trnsfm2 = trnsfmGray, class_num = 63)
    test_dset = ImageDataset(os.path.join(args.data_dir, 'test'), trnsfm1 =trnsfmColor, trnsfm2 = trnsfmGray, class_num = 63)
    
    train_loader = DataLoader(train_dset, batch_size=(args.bsize // args.bdivs), shuffle=True)

    alpha = torch.tensor(args.alpha).to(gpu)

    # ================================================
    # Training Phase 1
    # ================================================
    model_cn = CompletionNetwork()
    if args.data_parallel:
        model_cn = DataParallel(model_cn)
    if args.init_model_cn != None:
        model_cn.load_state_dict(torch.load(args.init_model_cn, map_location='cpu'))
    if args.optimizer == 'adadelta':
        opt_cn = Adadelta(model_cn.parameters())
    else:
        opt_cn = Adam(model_cn.parameters(),lr=0.5)
    model_cn = model_cn.to(gpu)

    writer = SummaryWriter()

    # training
    
    cnt_bdivs = 0
    pbar = tqdm(total=args.steps_1)
    while pbar.n < args.steps_1:
        for batch_index, (X, X_mask, X_inpaint, Labels) in enumerate(train_loader):
            # forward
            X = X.to(gpu)
            X_mask = X_mask.to(gpu)
            X_inpaint = X_inpaint.to(gpu)

            condition_g = label2img(
                shape=(X.shape[0],Labels.shape[2],X.shape[2]//2,X.shape[3]//2),
                labels = Labels
            )
            condition_g = condition_g.to(gpu)

            input = torch.cat((X_inpaint, X_mask), dim=1)
            output = model_cn(input,condition_g)

            loss = completion_network_loss(X, output, X_mask)
            
            # backward
            loss.backward()
            cnt_bdivs += 1

            # tensorboard用log出力設定1[ポイント3]
            writer.add_scalar('data/train_loss', loss.item(), batch_index)
            batch_index += 1

            torch.cuda.empty_cache()

            if cnt_bdivs >= args.bdivs:
                cnt_bdivs = 0
                # optimize
                opt_cn.step()
                # clear grads
                opt_cn.zero_grad()
                # update progbar
                pbar.set_description('phase 1 | train loss: %.5f' % loss.cpu())
                pbar.update()
                # test
                if pbar.n % args.snaperiod_1 == 0 or pbar.n == 1:
                    with torch.no_grad():

                        model_cn_path = os.path.join(args.result_dir, 'phase_1', 'model_cn_step%d' % pbar.n)

                        if args.data_parallel:
                            torch.save(model_cn.module.state_dict(), model_cn_path)
                        else:
                            torch.save(model_cn.state_dict(), model_cn_path)

                        completed = -1

                        # In order to avoid ROI error(regarding with OpenCV),
                        # when it comes to ROI error happened, poisson_blend returns "int" type

                        while(not isinstance(completed,torch.Tensor)):
                            X, X_mask,X_inpaint, Labels = sample_random_batch(test_dset, batch_size=args.num_test_completions)
                            
                            X = X.to(gpu)
                            X_mask = X_mask.to(gpu)
                            X_inpaint = X_inpaint.to(gpu)
                            condition_g = label2img(
                                shape=(X.shape[0],Labels.shape[2],X.shape[2]//2,X.shape[3]//2),
                                labels = Labels
                            )

                            condition_g = condition_g.to(gpu)

                            input = torch.cat((X_inpaint, X_mask), dim=1)
                            output = model_cn(input,condition_g)

                            completed = poisson_blend(X_inpaint, output, X_mask)

                        imgs = torch.cat((X.cpu(), X_inpaint.cpu(), completed.cpu()), dim=0)
                        imgpath = os.path.join(args.result_dir, 'phase_1', 'step%d.png' % pbar.n)
                        writer.add_image("generated",make_grid(imgs),pbar.n)
                        save_image(imgs, imgpath, nrow=len(X))
                       
                # terminate
                if pbar.n >= args.steps_1:
                    break
    pbar.close()
    writer.close()
    


    pbar.close()
    


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
