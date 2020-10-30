def main():

    import torch
    import os
    from torchvision import datasets, transforms
    import numpy as np
    import matplotlib.pyplot as plt
    from torchvision.utils import save_image,make_grid
    
    import sys
    sys.path.append('../')

    from datasets import ImageDataset
    from PIL import Image, ImageChops
    
    # ================================================
    # Test of making dataset
    # ================================================

    # dataset 
    size = (60,60)
    trnsfmColor = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])

    trnsfmGray = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize(size),
        transforms.ToTensor()
    ])

    print("loading dataset... (it may take a few minutes)")
    data_dir = os.path.join("/home/miyazonotaiga/デスクトップ/MyResearch/multidimensional_vector-estimation", 'data')
  
    #train_dset = ImageDataset(data_dir, trnsfm=trnsfmColor,class_num = 52)
    test_dset = ImageDataset(os.path.join(args.data_dir, 'test'), trnsfm, recursive_search=args.recursive_search)

    train_loader = torch.utils.data.DataLoader(train_dset, batch_size=4, shuffle=True)

    
    import cv2
    from tqdm import tqdm
    import time

    for images,labels,vectors in tqdm(train_loader):
        imgs = torch.cat((images,images), dim=0)
        print(vectors)
        imgpath = os.path.join(data_dir, 'step1.png')
        save_image(imgs, imgpath, nrow=len(images))
        break
    
    
if __name__ == '__main__':
    main()


